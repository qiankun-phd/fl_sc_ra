import torch
from marl_network import SharedActorCriticNetwork
from ppo import ppo_policy_data, ppo_policy_error
from gae import gae
from Env_SC import CustomWirelessEnv
from ReplayBuffer import ReplayBuffer
from typing import List


def sample_action(logit: List[torch.Tensor]) -> torch.Tensor:
    """
    **功能**:
        根据多个 logit 分布依次采样动作，并返回所有动作的张量。

    **输入**:
        - `logit`: 一个包含多个 `torch.Tensor` 的列表，每个张量表示某一动作维度的 logits。
                   每个张量的形状为 `(batch_size, agent_num, action_dim)`。

    **输出**:
        - 返回形状为 `(batch_size, agent_num, len(logit))` 的张量，表示所有维度的采样动作。
    """
    actions = []  # 存储每个维度采样的动作
    for logits in logit:
        prob = torch.softmax(logits, dim=-1)  # 计算概率
        dist = torch.distributions.Categorical(probs=prob)  # 创建分布
        actions.append(dist.sample())  # 对当前维度进行采样

    # 合并所有维度的动作到最后一维
    return torch.stack(actions, dim=-1)


def update(model, buffer, done) -> None:

    entropy_weight = 0.001
    # 价值损失权重，旨在平衡不同损失函数量级。
    value_weight = 0.5
    # 未来奖励的折扣系数。
    discount_factor = 0.99
    # 根据运行环境设置 tensor 设备为 cuda 或者 cpu。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义多智能体神经网络和优化器。
    model.to(device)
    # Adam 是深度强化学习中最常用的优化器。 如果你想添加权重衰减机制，应该使用 ``torch.optim.AdamW`` 。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义相应的测试数据，需要保持数据格式与环境交互生成的数据格式相同。
    # 注意，数据应该与网络保持相同的计算设备 (device)。
    # 为简单起见，这里我们将整个批次数据视为一个完整的 episode。
    # 在实际应用中，训练批次是多个 episode 的组合。我们通常使用 ``done`` 变量来划分不同的 episode 。

    state, action, reward, next_states, logit_old, value_old, return_ = buffer.get()

    # Actor-Critic 网络前向传播。
    output = model(state)
    # ``squeeze`` 操作将 shape 从 $$(B, A, 1)$$ 转化为 $$(B, A)$$.
    value = output['value'].squeeze(-1)
    # 使用广义优势估计（Generalized Advantage Estimation，简称GAE）方法来计算优势（Advantage）。
    # 优势是策略损失的一种“权重”，因此它被包含在 ``torch.no_grad()`` 中，表示不进行梯度计算。
    # ``done`` 是回合结束的标志。``traj_flag`` 是轨迹（trajectory）的标志。
    # 在这里，我们将整个批次数据视为一个完整的回合，所以 ``done`` 和 ``traj_flag`` 是相同的。
    with torch.no_grad():
        traj_flag = done
        gae_data = (value, value_old, reward, done, traj_flag)
        adv = gae(gae_data, discount_factor, 0.95)
    # 为 PPO policy loss 计算准备数据.
    data = ppo_policy_data(output['logit'], logit_old, action, adv, None)
    # 计算 PPO policy loss.
    loss, info = ppo_policy_error(data)
    # 计算 value loss.
    value_loss = torch.nn.functional.mse_loss(value, return_)
    # 策略损失 (PPO policy loss)、价值损失 (value loss) 和熵损失 (entropy_loss) 的加权和。
    total_loss = loss.policy_loss + value_weight * value_loss - entropy_weight * loss.entropy_loss

    # PyTorch loss 反向传播和优化器更新。
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # 打印训练信息。
    print(
        'total_loss: {:.4f}, policy_loss: {:.4f}, value_loss: {:.4f}, entropy_loss: {:.4f}'.format(
            total_loss, loss.policy_loss, value_loss, loss.entropy_loss
        )
    )
    print('approximate_kl_divergence: {:.4f}, clip_fraction: {:.4f}'.format(info.approx_kl, info.clipfrac))
    print('mappo_training_opeator is ok')



def train() -> None:
    batch_size, agent_num, obs_shape, n_cell, n_usr, n_channel, buffer_size = 4, 18, 108, 3, 18, 6, 200
    action_shape = [7, 7, 21]
    model = SharedActorCriticNetwork(agent_num, obs_shape, action_shape)
    env = CustomWirelessEnv(n_cell, n_usr, n_channel)
    state = env.reset()

    episode_reward, done = 0, False


    buffer = ReplayBuffer(buffer_size, obs_shape, action_shape)

    for ep in range(3000):
        for b in range(batch_size):
            output = model(state)
            action = sample_action(output['logit'])
            next_state, reward, done, info = env.step(action)
            buffer.store(state, action, reward, next_state, output['logit'], output['value'])
            state = next_state
            episode_reward += reward
        buffer.compute_returns_and_advantages(last_value=0)
        update(model, buffer, done)

if __name__ == "__main__":
    train()