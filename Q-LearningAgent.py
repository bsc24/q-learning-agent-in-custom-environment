import gym
import numpy as np
import os
import random
import sys
import time


class QAgent():
    def __init__(self, env, map_name=None, discount_rate=0.97, learning_rate=0.01, eps=1.0):
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        if (self.is_discrete):
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)

        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.eps = eps
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        if map_name is not None:
            self.load_model(map_name)
        else:
            self.build_model()

        # self.unexplored = np.array([[0]*self.action_size]*self.state_size, dtype=np.int8)

    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def load_model(self, map_name):
        file_location = "maps_and_qtables/" + map_name + "/qtable.txt"
        try:
            file = open(file_location)
        except FileNotFoundError:
            print("Could not find model matching provided name:" + map_name)
            print("Building new model instead")
            self.build_model()
            return
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
        # action_random = super().get_action(state)
        if (random.random() < self.eps):        # Generating a value less than the epsilon value results in a random action
            # return action_random
            actions = []
            for action in range(len(q_state)):
                if q_state[action] >= 0:
                    actions.append(action)

            return random.choice(actions)
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


env_name = 'maze_environment:BscMaze-v1'

start_state = 0
learn_rate = 0.01
epsilon_value = 1.0

# Checking to try and load a q-table from a provided file
if (len(sys.argv) > 1):
    map_name = sys.argv[-1]
    file_location = "maps_and_qtables/" + map_name + "/map.txt"
    try:
        file = open(file_location)
    except FileNotFoundError:
        print("Could not find map name matching provided name:" + map_name)
        exit(1)
    except:
        print("Some un-caught exception occurred in load_model()")
        exit(1)

    desc = []
    for line in file.readlines():
        if (line[0] == "s"):
            start_state = int(line[1:])
            continue
        elif (line[0:2] == "ep"):
            epsilon_value = float(line[2:])
            continue

        holder = ""
        for entry in line.split(',')[:-1]:
            holder += entry
        desc.append(holder)

    # epsilon_value = 0.0
    env = gym.make(env_name, start_state=start_state, desc=desc)
    # print("Starting at state: " + str(start_state))
    agent = QAgent(env, map_name, learning_rate=learn_rate, eps=epsilon_value)
    del(desc, map_name)
else:
    env = gym.make(env_name)
    agent = QAgent(env, learning_rate=learn_rate, eps=epsilon_value)


# This is used after agent is completed to write the q table to a file
time_started = time.time()

output = True       # Set this to False to skip print out of visualization
num_trials = 1000
total_attempts = 0
total_successes = 0
consecutive_success_counter = 0
actions = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}
# while True:
for trial in range(num_trials):
    state = env.reset()
    t = 0
    done = False
    current_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state, action, next_state, reward, done))
        state = next_state
        current_reward += reward

        if (output):
            print("Episode: {}, Episodes without incident: {}, Total successes: {}, Eps: {}"
                  .format(trial, consecutive_success_counter, total_successes, agent.eps))
            print("Current state: {}, Steps taken this run: {}, Last action: {}"
                  .format(state, t, actions[action]))

            env.render()
            # print(agent.q_table)      # Uncomment this line for q_table print out
            time.sleep(0.1)
            os.system('CLS')
        t += 1
        if (t == 100):
            done = True
            agent.eps = agent.eps * 0.99

    total_attempts += 1
    if done and reward == 10:
        consecutive_success_counter += 1
        total_successes += 1
    else:
        consecutive_success_counter = 0
    # print("Trial:", trial, "\nReward:", total_reward)
    if (consecutive_success_counter == 100):
        print("Agent has solved the puzzle.")
        break

os.system('CLS')
time_finished = time.time()
print("Total attempts by the agent: {}".format(total_attempts))
print("Total successes achieved by agent: {}".format(total_successes))
print("Time taken (in seconds): {}".format(time_finished-time_started))
print("Final Q-Table:")
print(agent.q_table)

time_name = time.localtime(time.time())
time_name = str(time_name[0]) + "_" \
               + str(time_name[2]) + "_" \
               + str(time_name[3]) + "_" \
               + str(time_name[4]) + "_" \
               + str(time_name[5])
directory = "maps_and_qtables/results/" + time_name
if (not os.path.isdir(directory)):
    os.mkdir(directory)

# Writing statistics from agent
file = open(directory + "/stats.txt", "w")
file.write("Total attempts by the agent: {}".format(total_attempts))
file.write("\nTotal successes achieved by agent: {}".format(total_successes))
file.write("\nTime taken (in seconds): {}".format(time_finished-time_started))
file.close()

# Writing q-table to file
file = open(directory + "/qtable.txt", "w")
q_table = agent.q_table
for row in q_table:
    for entry in row:
        file.write(str(entry) + ',')
    file.write("\n")
file.close()


# Writing map to file
file = open(directory + "/map.txt", 'w')
file.write("s" + str(start_state) + "\n")
file.write("ep" + str(agent.eps) + "\n")
for row in env.desc:
    for entry in row:
        file.write(entry.decode('UTF-8') + ',')
    file.write("\n")
file.close()


env.close()
