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
ACTIONS = {
    b'L':LEFT,
    b'D':DOWN,
    b'R':RIGHT,
    b'U':UP
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
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    NEW:
        U, D, R, L : Move either (U)p, (D)own, (R)ight, or (L)eft when this tile is stepped on
        C : Continue in direction; if moving down on this tile continue down again, if moving right on this tile continue right again, etc
        W : Wall, attempting to move into a tile with a wall results in nothing happening

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 10 if you reach the goal, and -1 for falling into a hole.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, start_state=0, is_slippery=False):

        if desc is None:           # If we aren't given a map name or desc, generate a random one
            desc = generate_random_map()          # desc now holds a list of strings where each string is a row in the map

        self.desc = desc = np.asarray(desc, dtype='c')          # desc is being turned from a list of strings into a numpy array
        self.start_state = start_state
        self.nrow, self.ncol = nrow, ncol = desc.shape          # self.nrow = nrow = desc.shape[0]      self.ncol = ncol = desc.shape[1]
        print(nrow, ncol)
        self.reward_range = (0, 1)

        nA = 4          # At each state, there are 4 possible actions (moving either left, down, right, or up)
        nS = nrow * ncol            # Number of states is the number of rows times number of columns (cuz it's a square... duh)

        isd = np.array(desc == b'S').astype('float64').ravel()          # Initial state distribution
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}          # P is transitions, it's a dictionary of lists where P[s][a] == [(probability, nextstate, reward, done), ...]

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
            if newletter == b'G':
                reward = 10
            elif newletter == b'H':
                reward = -1
            else:
                reward = 0
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
                        # if letter in b'LDRU':
                        #     li.append((
                        #         1., *update_probability_matrix(row, col, ACTIONS[letter])
                        #         ))
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
        self.s = self.start_state


    def step(self, a):
        transitions = self.P[self.s][a]
        # print(transitions)
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]

        row = s // self.nrow
        col = s - (row * self.nrow)
        letter = self.desc[row, col]

        while letter in b'CLDRU':
            last_s = s
            if letter in b'LDRU':
                a = ACTIONS[letter]
                p, s, r, d = self.P[s][a][0]
            elif letter == b'C':
                p, s, r, d = self.P[s][a][0]        # 'C' makes the agent continue in the direction it last moved in

            row = s // self.nrow
            col = s - (row * self.nrow)
            letter = self.desc[row, col]

            if (last_s == s):       # If we tried moving again but it put us at the same state, we tried to walk off the edge of the map and should stop trying
                break

        if letter == b'W':          # When we hit a wall, we walk into it then step back out
            reverse_a = (a + 2) % 4
            p, s, r, d = self.P[s][reverse_a][0]

        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})


    def reset(self):
        self.s = self.start_state
        self.lastaction = None
        return int(self.s)


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
