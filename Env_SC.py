import gym
from gym import spaces
import numpy as np
import math
from scipy.io import loadmat
from xxxxxxx import xxxx



class BaseStations:
    def __init__(self, n_cell, isd=1000):
        self.n_cell = n_cell
        self.isd = isd
        self.bs_positions = self._initialize_positions()

    def _initialize_positions(self):
        bs_positions = []

        if self.n_cell == 1:
            bs_positions.append((0, 0))  # 单个基站的位置
        else:
            bs_positions.append((0, 0))
            if self.n_cell >= 2:
                bs_positions.append((self.isd, 0))

            for i in range(2, self.n_cell):
                angle = 2 * math.pi * i / self.n_cell
                x = self.isd * math.cos(angle)
                y = self.isd * math.sin(angle)
                bs_positions.append((x, y))

        return bs_positions


class CustomWirelessEnv(gym.Env):
    def __init__(self, n_cell, n_usr, n_channel):
        super(CustomWirelessEnv, self).__init__()

        self.n_cell = n_cell
        self.n_usr = n_usr
        self.channels = n_channel

        self.cell_index = np.random.randint(1, self.n_cell + 1, size=self.n_usr)
        self.cell_num = np.zeros(self.n_cell, dtype=int)

        self.shadow_factor = 6
        self.bandwidth = 18000
        self.nr = 2
        self.nt = 1
        self.sinr_db = []

        self.bs_positions = BaseStations(n_cell).bs_positions
        self.user_positions = self._generate_user_distribution()

        self.simi = []
        self.w_phi = {}
        self.para_S = {}
        self.h = {}
        self.H_S = 4
        self.sem_table = loadmat('./DeepSC_table.mat')

        # 定义 action 和 observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(n_usr, 3), dtype=np.float32)
        state_dim = (n_cell * 4) + n_usr * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def _generate_user_distribution(self):
        for n in range(self.n_cell):
            self.cell_num[n] = np.sum(self.cell_index == n + 1)

        user_positions = []

        for i in range(self.n_cell):
            n_users = self.cell_num[i].item()
            user_pos = np.zeros((n_users, 2))

            radius_d = 1000 * np.sqrt(np.random.rand(n_users))
            phase = 2 * np.pi * np.random.rand(n_users)

            user_pos[:, 0] = radius_d * np.cos(phase) + self.bs_positions[i][0]
            user_pos[:, 1] = radius_d * np.sin(phase) + self.bs_positions[i][1]

            user_positions.append(user_pos)
        return user_positions

    def generate_user_parameters(self):
        for i in range(self.n_cell):
            n_users = int(self.cell_num[i])
            self.w_phi[i] = np.random.rand(n_users)

            phi_s = 50 + (70 - 50) * np.random.rand(n_users)
            beta_s = np.random.normal(0.2, 0.05, n_users)
            si_s = 0.8 + (0.9 - 0.8) * np.random.rand(n_users)
            lamda_s = np.random.normal(55, 2.5, n_users)

            self.para_S[i] = np.column_stack((beta_s, phi_s, lamda_s, si_s))

    def generate_channel_fading(self):
        h_large = {}  # 大规模衰落
        h_small = {}  # 小规模衰落

        for n_cell in range(self.n_cell):
            n_users = self.cell_num[n_cell]  # 当前基站的用户总数
            h_large[n_cell] = np.zeros((n_users, self.n_cell))  # 大规模衰落，从 n_cell 用户到所有基站

            # 计算大规模衰落
            for n_cell_1 in range(self.n_cell):
                d = np.sqrt((self.user_positions[n_cell][:, 0] - self.bs_positions[n_cell_1][0]) ** 2 +
                            (self.user_positions[n_cell][:, 1] - self.bs_positions[n_cell_1][1]) ** 2)  # 距离
                pl = 128.1 + 37.6 * np.log10(d / 1000)  # 路径损耗
                h_large[n_cell][:, n_cell_1] = 10 ** (-(pl + self.shadow_factor) / 10)  # 大规模衰落系数

            # 计算小规模衰落和综合信道系数
            for n_channel in range(self.channels):
                h_small[(n_cell, n_channel)] = np.zeros((self.n_cell, self.nr, n_users), dtype=complex)  # 小规模衰落
                self.h[(n_cell, n_channel)] = np.zeros((self.n_cell, self.nr, n_users), dtype=complex)  # 综合信道系数

                for n_d in range(n_users):  # 遍历 n_cell 的每个用户
                    h_real = np.random.randn(self.n_cell, self.nr) / np.sqrt(2)
                    h_imag = np.random.randn(self.n_cell, self.nr) / np.sqrt(2)
                    h_small[(n_cell, n_channel)][:, :, n_d] = h_real + 1j * h_imag  # 小规模衰落

                    for n_cell_1 in range(self.n_cell):  # 遍历所有基站
                        self.h[(n_cell, n_channel)][n_cell_1, :, n_d] = (
                                h_small[(n_cell, n_channel)][n_cell_1, :, n_d] *
                                np.sqrt(h_large[n_cell][n_d, n_cell_1])
                        )  # 综合信道系数

    def calculate_qoe_real_for_train(self, actions_all):
        channel_result = actions_all[:, 0]
        power_result = actions_all[:, 1]
        k_result = actions_all[:, 2]

        H_S = self.H_S
        G_th = 0.5
        w_phi = self.w_phi
        para_S = self.para_S
        bandwidth = self.bandwidth
        N_cell = self.n_cell
        h = self.h
        P_noise = 180000 * 10 ** (-17.4)

        sem_table = self.sem_table

        QoE_real = [None] * N_cell
        snr_range_S = np.arange(-10, 21, 1)  # SNR 范围

        k_range_S = np.arange(1, 21, 1)
        P_range_db = np.array([-10, -5, 0, 5, 10, 15, 20])
        P_range = 10 ** (P_range_db / 10)

        # 检查信道冲突
        total_channels = self.channels * self.n_cell
        conflicted_channels = 0

        cell_channels = defaultdict(list)
        cell_power = defaultdict(list)
        cell_seman = defaultdict(list)
        for i in range(self.n_usr):
            cell_index = self.cell_index[i]
            channel = channel_result[i]
            power = P_range[power_result[i]]
            k_s = k_range_S[k_result[i]]
            cell_channels[cell_index].append(channel)
            cell_power[cell_index].append(power)
            cell_seman[cell_index].append(k_s)

        for n_cell in range(N_cell):

            user_channels = cell_channels[n_cell + 1]
            user_channels = [ch for ch in user_channels if ch != 0]

            channel_counts = {}
            for ch in user_channels:
                channel_counts[ch] = channel_counts.get(ch, 0) + 1

            # 检测信道冲突
            for ch, count in channel_counts.items():
                if count > 1:
                    conflicted_channels += 1

        # 如果有信道冲突，返回冲突比例
        if conflicted_channels > 0:
            conflict_ratio = (conflicted_channels - total_channels) / total_channels
            return conflict_ratio

        # 如果没有信道冲突，计算 QoE
        for n_cell in range(N_cell):
            users = self.cell_num[n_cell]
            sinr_db = np.zeros(users)

            for n_d in range(users):
                if channel_result[n_cell][n_d] != 0:
                    I = 0
                    channel_i = channel_result[n_cell][n_d]
                    H = h[(n_cell, channel_i)][n_cell, :, n_d].T

                    # 计算干扰
                    for nn_cell in range(N_cell):
                        if nn_cell != n_cell:
                            interference_indices = np.where(channel_result[nn_cell] == channel_i)[0]
                            if interference_indices.size > 0:
                                in_d_i = interference_indices[0]
                                H_i = h[(nn_cell, channel_i)][n_cell, :, in_d_i].T
                                I += power[nn_cell][in_d_i] * (abs(H.conj().T @ H_i)) ** 2

                    # 计算 SINR
                    SINR = (power[n_cell][n_d] * (abs(H.conj().T @ H)) ** 2) / (
                            np.linalg.norm(H, 2) ** 2 * P_noise + I)
                    sinr_db[n_d] = round(10 * np.log10(SINR))
            self.sinr_db.append(sinr_db)

            # 单模用户 QoE 计算
            QoE_real[n_cell] = np.zeros(users)
            for n_s in range(users):
                if k_result[n_cell][n_s] != 0:
                    if sinr_db[n_s] < min(snr_range_S):
                        QoE_real[n_cell][n_s] = 0
                    else:
                        sinr_db[n_s] = min(max(sinr_db[n_s], min(snr_range_S)), max(snr_range_S))
                        sinr_index = int(sinr_db[n_s] - min(snr_range_S))
                        k_s = min(max(k_result[n_cell][n_s], min(k_range_S)), max(k_range_S))
                        k_index_s = int(k_s - min(snr_range_S))
                        si = sem_table[k_index_s, sinr_index]
                        phi = H_S / (k_result[n_cell][n_s] / bandwidth)
                        G_phi = 1 / (1 + math.exp(para_S[n_cell][n_s, 0] * (para_S[n_cell][n_s, 1] - phi / 1000)))
                        G_si = 1 / (1 + math.exp(para_S[n_cell][n_s, 2] * (para_S[n_cell][n_s, 3] - si)))
                        QoE = w_phi[n_cell][n_s] * G_phi + (1 - w_phi[n_cell][n_s]) * G_si
                        if G_si < G_th or G_phi < G_th:
                            QoE_real[n_cell][n_s] = 0
                        else:
                            QoE_real[n_cell][n_s] = QoE
            self.simi.append(sinr_db)
        return sum(QoE_real)

    def reset(self):
        self.generate_user_parameters()
        state_para_S = np.concatenate([self.para_S[i].flatten() for i in range(self.n_cell)])
        self.sinr_db = np.zeros(self.n_usr, dtype='int')
        self.simi = np.zeros(self.n_usr, dtype='float32')
        state = np.concatenate([state_para_S, self.sinr_db, self.simi])
        return state

    def step(self, action):
        reward = self.calculate_qoe_real_for_train(action)
        state_para_S = np.concatenate([self.para_S[i].flatten() for i in range(self.n_cell)])
        self.generate_user_parameters()
        next_state = np.concatenate([state_para_S, self.sinr_db, self.simi])
        done = False  # 根据任务定义终止条件
        info = {}  # 可选的附加信息
        return next_state, reward, done, info