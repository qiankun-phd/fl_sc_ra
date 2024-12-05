"""
PyTorch tutorial for neural networks in multi-agent cooperation scenarios, including independent actor-critic network (w or w/o shared parameters)
and centralized training critic network with decentralized execution (CTDE) actor-critic network. All the examples are based on the discrete action space.

This tutorial is mainly composed of three parts, you can learn from these parts in order or just jump to the part you are interested in:
  - Shared actor-critic network for independent agents
  - Independent actor-critic network for independent agents
  - CTDE actor-critic network for cooperation agents
More details about multi-agent cooperation reinforcement learning can be found in <link https://github.com/opendilab/PPOxFamily/blob/main/chapter6_marl/chapter6_lecture.pdf link>.
"""
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from typing import Dict


class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        super(ActorCriticNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        # Define discrete action logit output network, just one-layer FC.
        self.head = nn.ModuleList()
        for size in action_shape:
            self.head.append(nn.Linear(64, size))

        # Define scalar value output network.
        self.value_head = nn.Linear(64, 1)

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
        """
        **Overview**:
            The computation graph of actor-critic network in discrete action space.
        """
        # Transform original local obs into embedding vector, i.e. $$(B, A, *) -> (B, A, N)$$
        # Some network layers in PyTorch like ``nn.Linear`` can deal with any number of prefix dimensions, so we can just use it to process the whole multi-agent batch.
        x = self.encoder(local_obs)
        # Calculate logit for each possible discrete action choices, i.e. $$(B, A, N) -> (B, A, M)$$
        logit = [h(x) for h in self.head]
        # Calculate value for each sample and agent, i.e. $$(B, A, N) -> (B, A, 1)$$
        value = self.value_head(x)
        # Return the final result by treetensor format.
        return {'logit': logit, 'value': value}


# delimiter
class SharedActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of shared parameters actor-critic network in policy gradient algorithms for multi-agent scenarios.
            Each agent shares the same parameters in the network so that they can be processed as a batch in parallel.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(SharedActorCriticNetwork, self).__init__()
        # The shape of forward input is $$(B, A, O)$$.
        self.agent_num = agent_num
        # Define a shared actor-critic network used for all the agents.
        self.actor_critic_network = ActorCriticNetwork(obs_shape, action_shape)

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
        """
        **Overview**:
            The computation graph of shared parameters actor-critic network, processing all agents' ``local_obs`` and output
            corresponding policy logit and value respectively.
        """
        # Call the actor_critic_network in parallel.
        return self.actor_critic_network(local_obs)


# delimiter
class IndependentActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of independent actor-critic network in policy gradient algorithms for multi-agent scenarios.
            Each agent owns an independent actor-critic network with its own parameters.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(IndependentActorCriticNetwork, self).__init__()
        # Define ``agent_num`` independent actor-critic networks for each agent.
        # To reuse some attributes of ``nn.Module`` , we use ``nn.ModuleList`` to store these networks instead of Python native list.
        self.agent_num = agent_num
        self.actor_critic_networks = nn.ModuleList(
            [ActorCriticNetwork(obs_shape, action_shape) for _ in range(agent_num)]
        )

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
        """
        **Overview**:
            The computation graph of independent actor-critic network, serially processing each agent's
            ``local_obs`` and output the cooresponding policy logit and value respectively.
        """
        # Slice data, call the actor_critic_network serially, then concatenate the output.
        # 假设 result 是一个包含字典的列表
        result = [net(local_obs[:, i:i + 1]) for i, net in enumerate(self.actor_critic_networks)]

        # 分别合并 'logit' 和 'value'
        logits = [
            torch.cat([res['logit'][j] for res in result], dim=1)  # 在第一维度 (dim=1) 合并
            for j in range(len(result[0]['logit']))  # 遍历 logit 的每个子列表项
        ]
        values = torch.cat([res['value'] for res in result], dim=1)

        # 将结果重新组合为字典
        return {'logit': logits, 'value': values}


# delimiter
class CTDEActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, local_obs_shape: int, global_obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of centralized training decentralized execution (CTDE) actor-critic network in policy gradient algorithms for multi-agent scenarios.
            Each agent shares the same parameters in the network so that they can be processed as a batch in parallel.
            The input of value network is ``global_obs`` while the input of policy network is ``local_obs`` .
            Global information used in value network can provide more guidance for the training of policy network.
            Local information used in policy network can make the policy network more robust to the decentralized execution.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(CTDEActorCriticNetwork, self).__init__()
        # Define local and global encoder respectively.
        self.agent_num = agent_num
        self.local_encoder = nn.Sequential(
            nn.Linear(local_obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        # Define discrete action logit output network, just one-layer FC.
        self.head = nn.ModuleList()
        for size in action_shape:
            self.head.append(nn.Linear(64, size))
        # Define scalar value output network.
        self.value_head = nn.Linear(64, 1)

    # delimiter
    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
        """
        **Overview**:
            The computation graph of CTDE actor-critic network, processing all agents' ``local_obs`` and ``global_obs`` and output
            corresponding policy logit and value in parallel.
            There are two possible designs for ``global_obs`` : The former is a shared global state for all agents, i.e. $$(B, S)$$.
            Tha latter is a kind of agent-specific global state, i.e. $$(B, A, S')$$.
            For more details, you can refer to <link https://di-engine-docs.readthedocs.io/zh_CN/latest/04_best_practice/marl_zh.html#id10 link>.
        """
        # Call policy network with local obs and critic network with global obs respectively.
        policy = [h(self.local_encoder(local_obs)) for h in self.head]
        value = self.value_head(self.global_encoder(global_obs))
        return {
            'logit': policy,
            'value': value,}

def test_sample_multi_discrete_action(logit: torch.Tensor):
    # 对于多个 logit，依次调用采样函数，根据相应的 logit 采样得到最终的离散动作
    for i in range(len(logit)):
        prob = torch.softmax(logit[i], dim=-1)
        dist = torch.distributions.Categorical(probs=prob)
        action_i = dist.sample()
        print(action_i)


# delimiter
def test_shared_ac_network() -> None:
    """
    **Overview**:
        The function of testing shared parameters actor-critic network. Construct a network and pass a batch of data to it.
        Then validate the shape of different parts of output.
    """
    # Set batch size, agent number, observation shape and action shape.
    batch_size = 4
    agent_num = 18
    obs_shape = 108
    action_shape = [7, 7, 20]
    # Define a shared actor-critic network.
    network = SharedActorCriticNetwork(agent_num, obs_shape, action_shape)
    # Generate a batch of local obs data for all agents from the standard normal distribution.
    local_obs = torch.randn(batch_size, agent_num, obs_shape)
    # Actor-critic network forward procedure, pass the local obs data to the network and get the output.
    result = network(local_obs)
    logics = result['logit']
    # Validate the shape of output.


    # assert result['logit'].shape == (batch_size, agent_num, action_shape)
    test_sample_multi_discrete_action(result['logit'])

    assert result['value'].shape == (batch_size, agent_num, 1)


# delimiter
def test_independent_ac_network() -> None:
    """
    **Overview**:
        The function of testing independent actor-critic network. Construct a network and pass a batch of data to it.
        Then validate the shape of different parts of output.
    """
    # Set batch size, agent number, observation shape and action shape.
    batch_size = 4
    agent_num = 18
    obs_shape = 108
    action_shape = [7, 7, 20]
    # Define a independent actor-critic network.
    network = IndependentActorCriticNetwork(agent_num, obs_shape, action_shape)
    # Generate a batch of local obs data for all agents from the standard normal distribution.
    local_obs = torch.randn(batch_size, agent_num, obs_shape)
    # Actor-critic network forward procedure, pass the local obs data to the network and get the output.
    result = network(local_obs)
    # Validate the shape of output.
    # assert result['logit'].shape == (batch_size, agent_num, action_shape)
    test_sample_multi_discrete_action(result['logit'])
    assert result['value'].shape == (batch_size, agent_num, 1)


# delimiter
def test_ctde_ac_network() -> None:
    """
    **Overview**:
        The function of testing CTDE actor-critic network. Construct a network and pass a batch of data to it.
        Then validate the shape of different parts of output.
    """
    # Set batch size, agent number, observation shape and action shape.
    local_obs_shape = 5
    global_obs_shape = 108
    batch_size = 4
    agent_num = 18
    action_shape = [7, 7, 20]
    # Test case for the shared global obs.
    network = CTDEActorCriticNetwork(agent_num, local_obs_shape, global_obs_shape, action_shape)
    local_obs = torch.randn(batch_size, agent_num, local_obs_shape)
    global_obs = torch.randn(batch_size, global_obs_shape)
    result = network(local_obs, global_obs)
    # assert result['logit'].shape == (batch_size, agent_num, action_shape)
    test_sample_multi_discrete_action(result['logit'])
    assert result['value'].shape == (batch_size, 1)

    # Test case for the agent-specific global obs.
    agent_specific_global_obs_shape = 25
    network = CTDEActorCriticNetwork(agent_num, local_obs_shape, agent_specific_global_obs_shape, action_shape)
    local_obs = torch.randn(batch_size, agent_num, local_obs_shape)
    agent_specific_global_obs = torch.randn(batch_size, agent_num, agent_specific_global_obs_shape)
    result = network(local_obs, agent_specific_global_obs)
    # assert result['logit'].shape == (batch_size, agent_num, action_shape)
    test_sample_multi_discrete_action(result['logit'])
    assert result['value'].shape == (batch_size, agent_num, 1)




if __name__ == "__main__":
    test_shared_ac_network()
    test_independent_ac_network()
    test_ctde_ac_network()