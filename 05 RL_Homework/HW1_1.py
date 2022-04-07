import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



class bandit_setting(object):

    def __init__(self, number_arms, mean, variance):
        self.number_arms = number_arms
        self.mean = mean
        self.variance = variance
        self.action_value_set = np.zeros(self.number_arms)
        self.optim = 0
        self.reset()

    # Reset and a new game
    def reset(self):
        # randomly generated action_value_set
        self.action_value_set = np.random.normal(self.mean, self.variance, self.number_arms)
        # choose the action for max action_value
        self.optim = np.argmax(self.action_value_set)


class Agent(object):
    def __init__(self, number_arms, Epsilon):

        self.number_arms = number_arms
        self.Epsilon = Epsilon

        self.timeStep = 0
        self.lastAction = 0

        self.Action_times = np.zeros(self.number_arms)  # number of times a taken prior to t
        self.Sum_rewards = np.zeros(self.number_arms)  # Sum of rewards when a taken prior to t
        self.Q_t_a = np.zeros(self.number_arms)  # Q_t_a is the estimate value by averaging

    def choose_action(self):
        ep_rand = np.random.random()
        if ep_rand < self.Epsilon:
            a_out = np.random.choice(len(self.Q_t_a))  # random action
        else:
            maxAction = np.argmax(self.Q_t_a)  # Choose optim action
            # print(maxAction)
            # identify the corresponding action, as array containing only actions with max
            action = np.where(self.Q_t_a == maxAction)[0]
            # print(action)
            if len(action) == 0:
                a_out = maxAction
            else:
                a_out = np.random.choice(action)
        self.lastAction = a_out

        return a_out

    def calculate_Q_t_a(self, reward):

        self.Action_times[self.lastAction] += 1
        self.Sum_rewards[self.lastAction] += reward

        self.Q_t_a[self.lastAction] = self.Sum_rewards[self.lastAction] / self.Action_times[self.lastAction]

        self.timeStep += 1

    def reset(self):
        self.timeStep = 0  # Time Step t
        self.lastAction = None  # Store last action

        self.Action_times[:] = 0  # number of times a taken prior to t
        self.Sum_rewards[:] = 0  # Sum of rewards when a taken prior to t
        if self.Epsilon <= 0.0:
            self.Q_t_a[:] = 5  # for optimistic_agent + 5
        else:
            self.Q_t_a[:] = 0  # Q_t_a is the estimate value by averaging


number_arms = 10
iter_ = 2000
time_steps = 1000

bandit_ = bandit_setting(number_arms, mean=0, variance=1)
optimistic_agent = Agent(number_arms, Epsilon=0.0)
greedy_agent = Agent(number_arms, Epsilon=0.1)

optim_optimistic = np.zeros(time_steps)
optim_greedy = np.zeros(time_steps)

for i in range(iter_):
    bandit_.reset()
    greedy_agent.reset()
    optimistic_agent.reset()
    g_reward = 0
    o_reward = 0

    for j in range(time_steps):

        a = optimistic_agent.choose_action()
        reward_temp = np.random.normal(bandit_.action_value_set[a], scale=1)
        optimistic_agent.calculate_Q_t_a(reward=reward_temp)
        if a == bandit_.optim:
            optim_optimistic[j] += 1

        a = greedy_agent.choose_action()
        reward_temp = np.random.normal(bandit_.action_value_set[a], scale=1)
        greedy_agent.calculate_Q_t_a(reward=reward_temp)
        if a == bandit_.optim:
            optim_greedy[j] += 1

for i in range(time_steps):
    optim_optimistic[i] = optim_optimistic[i]/iter_
    optim_greedy[i] = optim_greedy[i] / iter_


time_array = np.arange(0, time_steps, 1)

plt.plot(time_array, optim_optimistic)
plt.plot(time_array, optim_greedy)
plt.xlabel("Steps", fontsize=13)
plt.ylabel("% Optimal action", fontsize=13)
plt.legend(["optimistic_agent", "greedy_agent"], loc=4)
plt.show()


# end of iter_




