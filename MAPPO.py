import tensorflow as tf
from tensorflow.keras import layers, Model

class ActorCriticNetwork(Model):

    def __init__(self, obs_shape: int, action_shape: int):
        super(ActorCriticNetwork, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])
        self.policy_head = layers.Dense(action_shape)
        self.value_head = layers.Dense(1)

    def call(self, local_obs):
        x = self.encoder(local_obs)
        logit = self.policy_head(x)
        value = self.value_head(x)
        return {'logit': logit, 'value': value}

class SharedActorCriticNetwork(Model):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int):
        super(SharedActorCriticNetwork, self).__init__()
        self.actor_critic_network = ActorCriticNetwork(obs_shape, action_shape)

    def call(self, local_obs):
        # Flatten the agent dimension into the batch
        batch_size, agent_num, obs_shape = tf.shape(local_obs)
        reshaped_obs = tf.reshape(local_obs, [-1, obs_shape])
        result = self.actor_critic_network(reshaped_obs)
        # Reshape back to (batch_size, agent_num, ...)
        logit = tf.reshape(result['logit'], [batch_size, agent_num, -1])
        value = tf.reshape(result['value'], [batch_size, agent_num, 1])
        return {'logit': logit, 'value': value}

class IndependentActorCriticNetwork(Model):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int):
        super(IndependentActorCriticNetwork, self).__init__()
        self.actor_critic_networks = [
            ActorCriticNetwork(obs_shape, action_shape) for _ in range(agent_num)
        ]

    def call(self, local_obs):
        outputs = [network(local_obs[:, i]) for i, network in enumerate(self.actor_critic_networks)]
        logits = tf.stack([output['logit'] for output in outputs], axis=1)
        values = tf.stack([output['value'] for output in outputs], axis=1)
        return {'logit': logits, 'value': values}

class CTDEActorCriticNetwork(Model):

    def __init__(self, agent_num: int, local_obs_shape: int, global_obs_shape: int, action_shape: int):
        super(CTDEActorCriticNetwork, self).__init__()
        self.local_encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])
        self.global_encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])
        self.power_policy_head = layers.Dense(action_shape[0])
        self.channel_policy_head = layers.Dense(action_shape[1])
        self.semantic_policy_head = layers.Dense(action_shape[2])
        self.value_head = layers.Dense(1)

    def call(self, local_obs, global_obs):
        power_logit = self.power_policy_head(self.local_encoder(local_obs))
        channel_logit = self.channel_policy_head(self.local_encoder(local_obs))
        semantic_logit = self.semantic_policy_head(self.local_encoder(local_obs))
        value = self.value_head(self.global_encoder(global_obs))
        return {'power_logit': power_logit, 'channel_logit': channel_logit,'semantic_logit': semantic_logit, 'value': value}

# Test Functions
def test_shared_ac_network():
    batch_size = 4
    agent_num = 3
    global_shape = 10
    local_shape = 5
    action_shape = [7, 7, 21]
    network = CTDEActorCriticNetwork(agent_num, global_shape, local_shape, action_shape)
    local_obs = tf.random.normal((batch_size, agent_num, local_shape))
    global_obs = tf.random.normal((batch_size, agent_num, global_shape))
    result = network(local_obs, global_obs)
    assert result['logit'].shape == (batch_size, agent_num, action_shape)
    assert result['value'].shape == (batch_size, agent_num, 1)

test_shared_ac_network()