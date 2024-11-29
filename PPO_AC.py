import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class PPO:
    def __init__(self, s_dim, c1, c2, epsilon, lr, K, n_veh, n_RB, n_Power, n_Seman):
        self.s_dim = s_dim
        self.a_dim = 3  # 动作维度：Power + RB_choice + Semantic_symbol
        self.K = K
        self.n_RB = n_RB
        self.n_Power = n_Power
        self.n_Seman = n_Seman
        self.n_veh = n_veh
        self.gamma = 0.99  # 折扣因子
        self.GAE_discount = 0.95  # GAE 参数
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2

        # 定义网络和优化器
        self.actor_critic = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.old_actor_critic = tf.keras.models.clone_model(self.actor_critic)

    def _build_net(self):
        """构建策略和价值网络"""
        inputs = tf.keras.Input(shape=(self.s_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        # 输出层：功率选择、资源块选择、语义符号选择
        Power_probs = tf.keras.layers.Dense(self.n_Power, activation='softmax')(x)

        Seman_probs = tf.keras.layers.Dense(self.n_Seman, activation='softmax')(x)

        RB_probs = tf.keras.layers.Dense(self.n_RB, activation='softmax')(x)
        v = tf.keras.layers.Dense(1)(x)

        # 定义模型
        model = tf.keras.Model(inputs=inputs, outputs=[Power_probs, Seman_probs, RB_probs, v])
        return model

    def choose_action(self, state):
        """从当前策略中选择动作"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        Power_probs, Seman_probs, RB_probs, _ = self.actor_critic(state)

        RB_action = tf.random.categorical(RB_probs, 1)  # 采样资源块
        Power_action = tf.random.categorical(RB_probs, 1)
        Seman_action = tf.random.categorical(RB_probs, 1)
        return [RB_action.numpy().flatten()[0], Power_action.numpy().flatten()[0], Seman_action.numpy().flatten()[0]]

    def compute_loss(self, states, actions, rewards, v_pred_next, gaes):
        """计算 PPO 损失函数"""
        RB_action = actions[0]
        Power_action = actions[1]
        Seman_action = actions[2]

        Power_probs, Seman_probs, RB_probs, v = self.actor_critic(states)
        RB_distribution = tfp.distributions.Categorical(probs=RB_probs)
        Power_distribution = tfp.distributions.Categorical(probs=Power_probs)
        Seman_distribution = tfp.distributions.Categorical(probs=Seman_probs)

        old_Power_probs, old_Seman_probs, old_RB_probs, _ = self.old_actor_critic(states)
        old_RB_distribution = tfp.distributions.Categorical(probs=old_RB_probs)
        old_Power_distribution = tfp.distributions.Categorical(probs=old_Power_probs)
        old_Seman_distribution = tfp.distributions.Categorical(probs=old_Seman_probs)



        ratio_RB = RB_distribution.prob(RB_action) / old_RB_distribution.prob(RB_action)
        ratio_Power = Power_distribution.prob(Power_action) / old_Power_distribution.prob(Power_action)
        ratio_Seman = Seman_distribution.prob(Seman_action) / old_Seman_distribution.prob(Seman_action)

        L_RB = tf.reduce_mean(tf.minimum(
            ratio_RB * gaes,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio_RB, 1 - self.epsilon, 1 + self.epsilon) * gaes
        ))

        L_Power = tf.reduce_mean(tf.minimum(
            ratio_Power * gaes,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio_Power, 1 - self.epsilon, 1 + self.epsilon) * gaes
        ))

        L_Seman = tf.reduce_mean(tf.minimum(
            ratio_Seman * gaes,
            tf.clip_by_value(ratio_Seman, 1 - self.epsilon, 1 + self.epsilon) * gaes
        ))

        # Value function loss
        L_vf = tf.reduce_mean(tf.square(rewards + self.gamma * v_pred_next - v))

        # Entropy bonus
        entropy = tf.reduce_mean(Power_distribution.entropy() + RB_distribution.entropy() + Seman_distribution.entropy())
        # 最终 PPO 损失
        loss = L_RB + L_Power + L_Seman - self.c1 * L_vf + self.c2 * entropy
        return loss

    def train(self, states, actions, rewards, v_pred_next, gaes):
        """更新策略网络"""
        for _ in range(self.K):  # K epochs
            with tf.GradientTape() as tape:
                loss = self.compute_loss(states, actions, rewards, v_pred_next, gaes)
                print(loss)
            grads = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

        # 更新旧网络参数
        self.old_actor_critic.set_weights(self.actor_critic.get_weights())

if __name__ == '__main__':
    ppo = PPO(126, 0.1, 0.2, 100, 0.05, 10, 9, 8, 7, 21)
    state = np.zeros(126, dtype=np.float32)
    state = np.expand_dims(state, axis=0)
    actions = np.zeros(3, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.float32)
    v_pred_next = np.zeros(1, dtype=np.float32)
    g = np.zeros(1, dtype=np.float32)
    ppo.train(state, actions, rewards, v_pred_next, g)
