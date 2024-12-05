import torch
from marl_network import CTDEActorCriticNetwork
from ppo import ppo_policy_data, ppo_policy_error
from gae import gae


# delimiter
def mappo_training_opeator() -> None:
    batch_size, agent_num, local_state_shape, agent_specific_global_state_shape = 4, 18, 5, 108
    action_shape = [7, 7, 21]
    # 熵加成权重，有利于探索。
    entropy_weight = 0.001
    # 价值损失权重，旨在平衡不同损失函数量级。
    value_weight = 0.5
    # 未来奖励的折扣系数。
    discount_factor = 0.99
    # 根据运行环境设置 tensor 设备为 cuda 或者 cpu。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义多智能体神经网络和优化器。
    model = CTDEActorCriticNetwork(agent_num, local_state_shape, agent_specific_global_state_shape, action_shape)
    model.to(device)
    # Adam 是深度强化学习中最常用的优化器。 如果你想添加权重衰减机制，应该使用 ``torch.optim.AdamW`` 。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义相应的测试数据，需要保持数据格式与环境交互生成的数据格式相同。
    # 注意，数据应该与网络保持相同的计算设备 (device)。
    # 为简单起见，这里我们将整个批次数据视为一个完整的 episode。
    # 在实际应用中，训练批次是多个 episode 的组合。我们通常使用 ``done`` 变量来划分不同的 episode 。
    local_state = torch.randn(batch_size, agent_num, local_state_shape).to(device)
    agent_specific_global_state = torch.randn(batch_size, agent_num, agent_specific_global_state_shape).to(device)
    logit_old = [
        torch.randn(batch_size, agent_num, action_dim).to(device)
        for action_dim in action_shape
    ]
    value_old = torch.randn(batch_size, agent_num).to(device)
    done = torch.zeros(batch_size).to(device)
    done[-1] = 1
    action = torch.stack([
        torch.randint(0, dim, (batch_size, agent_num)).to(device)
        for dim in action_shape
    ], dim=-1)
    reward = torch.randn(batch_size, agent_num).to(device)
    # 目标回报可以用不同的方法计算。这里我们使用奖励的折扣累计值。
    # 还可以使用广义优势估计 (GAE) 法、n-step TD 方法等等。
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-Critic 网络前向传播。
    output = model(local_state, agent_specific_global_state)
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


if __name__ == "__main__":
    mappo_training_opeator()