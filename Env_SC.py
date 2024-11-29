import gym
from gym import spaces
import numpy as np
import math
from scipy.io import loadmat
from collections import defaultdict


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
        # 信道衰落生成代码，与原始实现相同
        pass

    def calculate_qoe_real_for_train(self, actions_all):
        # QoE计算逻辑，保留原始实现
        pass

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