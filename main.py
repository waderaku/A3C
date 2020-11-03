import gym
from actor_critic import ActorCritic
from agent import Agent
import threading
from multiprocessing import cpu_count
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 学習定数
LEARNING_RATE = 5e-3
RMSPropDecaly = 0.99

# 割引率
GAMMA = 0.99

# 各スレッドの更新ステップ間隔
T_MAX = 10

# ε-greedyのパラメータ
EPS_START = 0.5
ANNEAL_RATE = 0.99

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env_name = env_name
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    global_actor_critic = ActorCritic(state_dim, action_dim)

    Agents = []
    num_agents = cpu_count()
    lock = threading.Lock()

    for i in range(num_agents):
        Agents.append(Agent(
            env_name,
            global_actor_critic,
            lock,
            EPS_START,
            ANNEAL_RATE,
            T_MAX,
            GAMMA,
            LEARNING_RATE
        ))

    for agent in Agents:
        agent.start()

    for worker in Agents:
        worker.join()
