import numpy as np
import math


class Env(object):
    def __init__(self):
        self.states = np.zeros((4, 4),dtype=float)
        self.new_states = np.zeros((4, 4),dtype=float)

        self.goal = (0, 0)
        self.goal2 = (3, 3)

        self.states[self.goal] = 100
        self.states[self.goal2] = 100

        self.new_states[self.goal] = 100
        self.new_states[self.goal2] = 100

    def init(self):
        self.new_states = np.zeros((4, 4))
        self.goal = (0, 0)
        self.goal2 = (3, 3)
        self.new_states[self.goal] = 100
        self.new_states[self.goal2] = 100

    def step(self):
        self.states = self.new_states.copy()
        self.init()
        r = -1
        for i in range(4):
            for j in range(4):
                if self.states[i][j] != 100:

                    try:  # up
                        if i-1 < 0:
                            self.new_states[i][j] += 0.25 * (r + self.states[i][j])
                        else:
                            if self.states[i - 1][j] == 100:
                                self.new_states[i][j] += 0.25 * r  # terminal
                            else:
                                self.new_states[i][j] += 0.25 * (r + self.states[i - 1][j])  # regular
                    except IndexError:
                        self.new_states[i][j] += 0.25 * (r + self.states[i][j])

                    try:  # down
                        if self.states[i+1][j] == 100:
                            self.new_states[i][j] += 0.25 * r
                        else:
                            self.new_states[i][j] += 0.25 * (r + self.states[i+1][j])
                    except IndexError:
                        self.new_states[i][j] += 0.25 * (r + self.states[i][j])

                    try:  # left
                        if j - 1 < 0:
                            self.new_states[i][j] += 0.25 * (r + self.states[i][j])
                        else:
                            if self.states[i][j-1] == 100:
                                self.new_states[i][j] += 0.25 * r
                            else:
                                self.new_states[i][j] += 0.25 * (r + self.states[i][j-1])
                    except IndexError:
                        self.new_states[i][j] += 0.25 * (r + self.states[i][j])

                    try:  # right
                        if self.states[i][j+1] == 100:
                            self.new_states[i][j] += 0.25 * r
                        else:
                            self.new_states[i][j] += 0.25 * (r + self.states[i][j+1])
                    except IndexError:
                        self.new_states[i][j] += 0.25 * (r + self.states[i][j])

                    self.new_states[i][j] = math.ceil(self.new_states[i][j] * 10) / 10.0


env = Env()
k_ = 20
print(env.states)
for i in range(1, 21):
    env.step()
    print("k =", i, )
    print(env.new_states)
