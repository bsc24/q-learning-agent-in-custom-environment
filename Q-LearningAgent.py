import gym
import numpy as np
import os
import random
import sys
import time


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
        if (self.is_discrete):
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01, q_table_file=None):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        if (q_table_file is not None):
            self.load_model(q_table_file)
        else:
            self.build_model()

    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def load_model(self, file_location):
        try:
            file = open(file_location)
        except FileNotFoundError:
            print("Could not find map name matching provided name:" + map_name)
            print("Building new model instead")
            self.build_model()
        except:
            print("Some un-caught exception occurred in load_model()")
            exit(1)

        table = []
        for line in file.readlines():
            actions_values = line.split(",")[:4]    # When saved, the q-table states have an extra comma (,) at the end of each line
            table.append(np.array(actions_values, dtype=np.float64))

        self.q_table = np.array(table)

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


env_name = 'maze_environment:BscMaze-v0'
env = gym.make(env_name)

# Checking to try and load a q-table from a provided file
if (len(sys.argv) > 1):
    map_name = sys.argv[1]
    directory = "maps_and_qtables/" + map_name + "/qtable.txt"
    agent = QAgent(env, directory, learning_rate=0.1)
else:
    agent = QAgent(env, learning_rate=0.1)


# This is used after agent is completed to write the q table to a file
time_made = time.localtime(time.time())
time_made = str(time_made[0]) + "_" \
            + str(time_made[2]) + "_" \
            + str(time_made[3]) + "_" \
            + str(time_made[4]) + "_" \
            + str(time_made[5])


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


# Writing q-table to file
directory = "maps_and_qtables/" + time_made
if (not os.path.isdir(directory)):
    os.mkdir(directory)
file = open(directory + "/qtable.txt", "w")
q_table = agent.q_table
print(type(q_table))
for row in q_table:
    print(type(row))
    for entry in row:
        file.write(str(entry) + ',')
        print(type(entry))
    file.write("\n")

file.close()

env.close()
