import gym
from actor_critic import ActorCritic
import numpy as np
import tensorflow as tf
from threading import Thread


class Agent(Thread):
    def __init__(
        self,
        env_name,
        global_actor_critic,
        lock,
        eps=0.5,
        anneal_rate=0.99,
        t_max=10,
        gamma=0.99,
        lr=0.0005
    ):
        Thread.__init__(self)
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.lock = lock
        self.eps = eps
        self.anneal_rate = anneal_rate
        self.global_actor_critic = global_actor_critic
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.t_max = t_max
        self.gamma = gamma

        self.optimizer = tf.keras.optimizers.Adam(lr)
        log_dir = 'logs'
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def get_action(self, policy):
        rand = np.random.rand()
        self.eps = self.eps * self.anneal_rate
        # ランダム行動
        if self.eps > rand:
            return np.random.randint(0, 2)
        # モデルに従った確率行動
        else:
            return np.random.choice(self.action_dim, p=policy[0])

    def train(self, max_episodes=100000):
        for i in range(max_episodes):
            # 重み行列の更新
            self.actor_critic.set_weights(
                self.global_actor_critic.get_weights())

            # env初期化
            episode_reward = 0
            done = False
            state = self.env.reset()
            shape = (1,) + state.shape
            state = state.reshape(shape)

            state_list = []
            action_list = []
            reward_list = []

            t = 0
            # ゲーム開始
            while not done:
                policy, value = self.actor_critic(state)
                next_action = self.get_action(policy.numpy())
                next_state, reward, done, _ = self.env.step(next_action)

                state = state.reshape(shape)
                action = np.reshape(next_action, [1, 1])
                next_state = next_state.reshape(shape)
                reward = np.reshape(reward, [1, 1])

                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)

                # グローバルモデルの更新
                if t == self.t_max or done:
                    t = 0
                    state_batch = self.list_to_batch(state_list)
                    action_batch = self.list_to_batch(action_list)
                    reward_batch = self.list_to_batch(reward_list)

                    _, next_v_value = self.actor_critic(
                        next_state)
                    td_target_batch = self.n_step_td_target(
                        reward_batch, next_v_value, done)
                    advatnage_batch = td_target_batch - \
                        self.actor_critic(state_batch)[1]

                    with self.lock:
                        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                            logits, value = self.global_actor_critic(
                                state_batch)
                            actor_loss, critic_loss = self.global_actor_critic.calc_loss(
                                state_batch, action_batch, advatnage_batch, logits, value, td_target_batch)

                        # NN更新
                        critic_grads = actor_tape.gradient(
                            critic_loss, self.global_actor_critic.trainable_variables)
                        self.optimizer.apply_gradients(
                            zip(critic_grads, self.global_actor_critic.trainable_variables))

                        actor_grads = critic_tape.gradient(
                            actor_loss, self.global_actor_critic.trainable_variables)

                        self.optimizer.apply_gradients(
                            zip(actor_grads, self.global_actor_critic.trainable_variables))

                        self.actor_critic.set_weights(
                            self.global_actor_critic.get_weights())
                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []
                state = next_state
                t += 1
                episode_reward += reward[0][0]
            with self.summary_writer.as_default():
                tf.summary.scalar('Reward', episode_reward, step=i)

    def run(self):
        self.train()
