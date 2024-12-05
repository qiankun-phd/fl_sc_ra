import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int, state_shape: int, action_shape: int, gamma: float = 0.99, lamda: float = 0.95):
        """
        **初始化 PPO Buffer**:
            用于存储与每个时间步相关的数据，用于 PPO 更新。

        **参数**:
        - `buffer_size`: Buffer 的容量（每个 episode 的最大时间步数）。
        - `state_shape`: 状态的形状。
        - `action_shape`: 动作的形状。
        - `gamma`: 折扣因子，用于计算回报（return）。
        - `lamda`: 用于计算优势的 Lambda 值（广义优势估计的系数）。
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lamda = lamda

        # 用于存储数据的列表
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []

        # 计算返回值
        self.returns = []

        # 当前缓存的数据长度
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, state, action, reward, next_state, log_prob, value):
        """
        **存储一个时间步的数据**:
            将每个时间步的状态、动作、奖励、下一个状态、log概率、值函数存储到 buffer 中。

        **参数**:
        - `state`: 当前时间步的状态。
        - `action`: 当前时间步的动作。
        - `reward`: 当前时间步的奖励。
        - `next_state`: 下一时间步的状态。
        - `log_prob`: 当前时间步的 log 概率。
        - `value`: 当前时间步的值函数。
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor):
        """
        **计算回报和优势**:
            根据折扣因子 `gamma` 和广义优势估计（GAE）公式计算回报和优势。

        **参数**:
        - `last_value`: 最后一个时间步的值函数。
        """
        # 计算回报
        returns = torch.zeros(self.ptr, dtype=torch.float32)
        advantages = torch.zeros(self.ptr, dtype=torch.float32)

        # 初始的返回值是最后一个状态的值函数
        next_value = last_value
        prev_advantage = 0

        for t in reversed(range(self.ptr)):
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            advantages[t] = prev_advantage = delta + self.gamma * self.lamda * prev_advantage
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        self.returns = returns

    def get(self):
        """
        **获取 buffer 中的数据**:
            返回所有存储的数据并清空 buffer。
        """
        # 转换为 tensor 格式
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        returns = torch.stack(self.returns)

        # 清空缓存
        self.clear()

        return states, actions, rewards, next_states, log_probs, values, returns

    def clear(self):
        """
        **清空 buffer**:
            清空当前存储的所有数据。
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.ptr = 0
        self.path_start_idx = 0