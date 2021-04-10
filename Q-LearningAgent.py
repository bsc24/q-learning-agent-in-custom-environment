import time

import gym
import random
import os
import numpy as np
from IPython.display import clear_output
from gym.envs.registration import register

try:
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery':False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    pass


class Agent():
    def __init__(self, env):
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        if (self.is_discrete):
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)

    def get_action(self, state):
        # cart_position = state[0]
        # cart_velocity = state[1]
        # pole_angle = state[2]
        # pole_velocity = state[3]
        # number = cart_position + cart_velocity + pole_angle + pole_velocity
        #
        # if (number < 0):
        #     action = 0
        # else:
        #     action = 1
        if (self.is_discrete):
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        if (random.random() < self.eps):
            return action_random
        else:
            return action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        if (done):
            q_next = np.zeros([self.action_size])

        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.eps = self.eps * 0.99


# env_name = 'CartPole-v0'
# env_name = 'MountainCarContinuous-v0'
# env_name = 'Acrobot-v1'
# env_name = "FrozenLakeNoSlip-v0"
# env_name = "FrozenLake-v0"
env_name = 'maze_environment:BscMaze-v0'
env = gym.make(env_name)

# agent = Agent(env)
agent = QAgent(env, learning_rate=0.1)

rewards_list = []
debug = False
num_trials = 1000
total_reward = 0
rewardCounter = 0
# while True:
for trial in range(num_trials):
    state = env.reset()
    t = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward += reward

        print("s:", state, "a:", action)
        print("Episode: {}, Total reward: {}, Eps: {}".format(trial, total_reward, agent.eps))
        env.render()
        print(agent.q_table)
        time.sleep(0.1)
        #clear_output(wait=True)
        os.system('CLS')

    if (reward == 1):
        rewardCounter += 1
    else:
        rewardCounter = 0
    rewards_list.append(total_reward)
    # print("Trial:", trial, "\nReward:", total_reward)
    if (rewardCounter == 100):
        break

os.system('CLS')
print("Agent has solved the puzzle.")
print("Total reward achieved by agent: {}".format(total_reward))
print("Final Q-Table:")
print(agent.q_table)



env.close()
