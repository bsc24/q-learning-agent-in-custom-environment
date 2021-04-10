import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# This environment was copied directly from the frozen_lake environment of OpenAI Gym
# For my final project, I intend to expand upon it by modifying things such as:
#     size of the grid
#     tiles which may cause different effects
#     changes to goal states and "win" conditions, such as:
#         multiple goal states in the map
#         must go to one point then another point and so on
#         must get to a point then, once it is considered completed (it has gone x number of trials without failing), change position of the goal state
#

MAPS = {
    "4x4": [        # "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"]
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [        # "8x8": ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class BscMazeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=False):
        if desc is None and map_name is None:           # If we aren't given a map name or desc, generate a random one
            desc = generate_random_map()
        elif desc is None:          # Else if we have a map name, load up the map that we have under that name
            desc = MAPS[map_name]           # desc now holds a list of strings where each string is a row in the map
        self.desc = desc = np.asarray(desc, dtype='c')          # desc is being turned from a list of strings into a numpy array
        self.nrow, self.ncol = nrow, ncol = desc.shape          # self.nrow = nrow = desc.shape[0]      self.ncol = ncol = desc.shape[1]
        self.reward_range = (0, 1)

        nA = 4          # At each state, there are 4 possible actions (moving either left, down, right, or up)
        nS = nrow * ncol            # Number of states is the number of rows times number of columns (cuz it's a square... duh)

        isd = np.array(desc == b'S').astype('float64').ravel()          # Initial state distribution
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}          # P is transitions, it's a dictionary of lists where P[s][a] == [(probability, nextstate, reward, done), ...]

        # The above is equivalent to:
        # P = {}
        # for state in range(nS):
        #     s = {}
        #     for action in range(nA):
        #         s[action] = []
        #     P[state] = s

        def to_s(row, col):         # Gets the state number from the provided (row, col)
            return row*ncol + col

        def inc(row, col, a):           # Used to increment row and col based on what action is provided
            if a == LEFT:           # The use of min() and max() in this function is to ensure that the index doesn't go out of bounds
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):            # Given a position and action, returns a tuple of state, reward, and whether the episode is done
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):          # range(4) because 4 actions per state (up, down, left, right)
                    li = P[s][a]            # li is a list of tuples of (odds of getting resulting state, resulting state) from the current state s with action a
                    letter = desc[row, col]         # Gets the letter being held at the row, col we are looking at ('S'=start, 'F'=empty, 'H'=hole, 'G'=goal)
                    if letter in b'GH':         # If we reach the Goal or Hole, we are done
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:     # if the ice is slippery, an action has an equal chance of going either the right way or in either of the two adjacent directions
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:         # ^For example, action down is equally likely to go left, right, or down; going right is equally likely to go up, down, or right
                                li.append((
                                    1. / 3.,            # 1/3 chance
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:       # Else, each action always goes in the intended direction
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(BscMazeEnv, self).__init__(nS, nA, P, isd)         # nS = number of states, nA = number of actions, P = transitions, isd = initial state distribution

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
