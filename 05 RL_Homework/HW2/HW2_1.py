'''
State:  ^0^ _ 1 _ 2 _ [3] _ 4 _ 5 _ ^6^
3: Start
0: Left
6: Right
'''
import random
import numpy as np
import matplotlib.pyplot as plt


class td_method_func(object):
    def __init__(self, Alpha):
        self.values = [0.5] * 7
        self.values[0] = 0
        self.values[6] = 1
        self.alpha = Alpha
        self.discount = 1.0

    def update(self, t_state, state, reward_t):
        reward_t = 0
        G_t = reward_t + self.discount * self.values[state]
        self.values[t_state] = self.values[t_state] + self.alpha * (G_t - self.values[t_state])

    def reset(self):
        self.values = [0.5] * 7
        self.values[0] = 0
        self.values[6] = 1

    def cal_RMS(self):
        true_value = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]
        num = 0
        for i in range(1, 6):
            num += (self.values[i] - true_value[i]) ** 2
        num = (1 / 5) * num
        rms = num ** 0.5

        return rms


class mc_method_func(object):
    def __init__(self, Alpha):
        self.values = [0.5] * 7
        self.values[0] = 0
        self.values[6] = 1
        self.alpha = Alpha

    def update(self, state_list, reward_t):
        # print("State_list: ", state_list)
        for i in state_list[:-1]:
            # MC update
            self.values[i] = self.values[i] + self.alpha * (reward_t - self.values[i])

    def reset(self):
        self.values = [0.5] * 7
        self.values[0] = 0
        self.values[6] = 1

    def cal_RMS(self):
        true_value = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]
        num = 0
        for i in range(1, 6):
            num += (self.values[i] - true_value[i]) ** 2
        num = (1 / 5) * num
        rms = num ** 0.5

        return rms


class environment(object):
    def __init__(self):
        self.pos = 3
        self.step = 0
        self.done = False
        self.reward = 0
        self.state_list = [3]

    def reset(self):
        self.pos = 3
        self.step = 0
        self.done = False
        self.reward = 0
        self.state_list = [3]
        # self.list_ = [0] * 5

    def random_walk(self):
        walk_direction = random.randint(0, 1)  # 0: left, 1: right
        if walk_direction == 0:
            self.pos -= 1
        else:
            self.pos += 1
        self.state_list.append(self.pos)
        self.step += 1
        # End or not
        if self.pos == 6:
            self.reward = 1
            self.done = True
        elif self.pos == 0:
            self.reward = 0
            self.done = True


env = environment()

############################################  TD ###################################################
td_alpha_list = [0.15, 0.1, 0.05]

for alpha in td_alpha_list:
    run = 100
    episodes = 100 + 1
    total_errors = np.zeros(episodes)
    td_method = td_method_func(Alpha=alpha)
    for i in range(run):
        env.reset()
        td_method.reset()
        errors = []
        for i in range(0, episodes):
            env.reset()
            rms = td_method.cal_RMS()
            errors.append(rms)
            # print("{} steps: {}".format(i, rms))
            # print("State value: ", td_method.values)
            while True:
                if env.done is True:
                    break
                else:
                    t_state = env.pos
                    env.random_walk()
                    # print("State : ", env.pos)
                    td_method.update(t_state, env.pos, env.reward)
        total_errors += np.asarray(errors)
    total_errors /= run
    # print("Total errors: ", total_errors)
    plt.plot(total_errors, label='TD' + ', alpha = {}'.format(td_method.alpha))

############################################  MC ###################################################

mc_alpha_list = [0.01, 0.02, 0.03, 0.04]
for alpha in mc_alpha_list:
    run = 100
    episodes = 100 + 1
    total_errors = np.zeros(episodes)
    mc_method = mc_method_func(Alpha=alpha)
    for i in range(run):
        env.reset()
        mc_method.reset()
        errors = []
        for i in range(0, episodes):
            env.reset()
            rms = mc_method.cal_RMS()
            errors.append(rms)
            # print("{} steps: {}".format(i, rms))
            # print("State value: ", mc_method.values)
            while True:
                if env.done is True:
                    mc_method.update(env.state_list, env.reward)
                    break
                else:
                    env.random_walk()
                    # print("State : ", env.pos)
        total_errors += np.asarray(errors)
    total_errors /= run
    # print("Total errors: ", total_errors)
    plt.plot(total_errors, label='MC' + ', alpha = {}'.format(mc_method.alpha))

plt.yticks(np.arange(0, 0.26, 0.05))
plt.xlabel('Walks / Episodes')
plt.legend()
plt.show()
