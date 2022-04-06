import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gurobi_func as gb

'''
    1000: Vehicle request sets
    If MEMORY_CAPACITY == 2000:
        It only run two random start_states
'''

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
# parameters
input_dim = 794
output_dim = 1
hidden_dim = 16
batch_size = 500
epochs = 5

Big_timer = 0

N_S_A = 794
s = [0 for i in range(794)]

env_state = []


def getWeightandBias(model):
    num = 0
    for par in model.parameters():
        if num == 0:
            fc1_weight = par.cpu().detach().numpy()
        elif num == 1:
            fc1_bias = par.cpu().detach().numpy()
        elif num == 2:
            fc2_weight = par.cpu().detach().numpy()
        else:
            fc2_bias = par.cpu().detach().numpy()
        num += 1
        # 0: fc1.weight: torch.Size([16, 794])
        # 1: fc1.bias: torch.Size([16])
        # 2: fc2.weight: torch.Size([1, 16])
        # 3: fc2.bias: torch.Size([1])
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


def getBigmap(i):
    if i % 25 == 0:
        i = i + 1
    loc_temp = int(i / 25)
    line_temp = int(i / 49)
    loc = int(i - 25 * loc_temp)
    if loc <= 0:
        ans = int(i / 2)
    else:
        map_loc = int(loc / 2)
        ans = int(map_loc + 13 * line_temp)
    return ans


def get_state_action():
    gen_state = np.random.choice([0, 1], size=(794), p=[0.65, 0.35])
    count = 0
    for i in gen_state:
        if i == 1:
            count += 1
    while count > 200:  # Normally it will not need this.
        rand_pos = np.random.randint(0, 794)
        if gen_state[rand_pos] == 1:
            gen_state[rand_pos] = 0
            count -= 1
        else:
            continue
    return gen_state


def generate_rand_action(s):
    state = s
    D_size = 1000
    while D_size > 200:
        count_ = 0
        action = get_state_action()
        for i in range(794):
            if action[i] - state[i] == 1:
                count_ += 1
        D_size = count_

    return action


class environment(object):
    def __init__(self, vec_num):
        self.states = None
        self.vec_num = vec_num
        self.timer = 0
        self.done = False

    def reset(self):
        self.states = None
        self.timer = 0
        self.done = False

    def getStartstate(self):
        self.states = get_state_action()
        return self.states

    def step(self, a):
        self.states = a  # Since our action and state is the same format, I change them directly.
        #  self.states is the s_ now.
        r = self.getReward()
        if self.timer >= 999:
            self.done = True
        else:
            self.timer += 1
        return self.states, r, self.done

    def getReward(self):
        reward_temp = 0
        for j in range(625):  # From map j in time i
            if self.states[j] == 1:  # Catched by Small m
                continue
            else:
                if self.states[625 + getBigmap(j)] == 1:  # Catched by Big M
                    reward_temp += -1.0 * self.vec_num[self.timer][j]
                else:  # Fail, and fetching from data set
                    reward_temp += -10.0 * self.vec_num[self.timer][j]

        return reward_temp


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_dim)
        # self.sigmoid = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        actions_value = self.relu(output)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(input_dim, hidden_dim, output_dim, batch_size), Net(input_dim, hidden_dim, output_dim,
                                                                                        batch_size)
        self.Big_timer = 0
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_S_A * 2 + 1 + N_S_A))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # Change to ILP Solver
    def choose_action(self, observation_state):
        # input only one sample
        fc1_weight, fc1_bias, fc2_weight, fc2_bias = getWeightandBias(dqn.target_net)
        gurobi_func = gb.Gurobi(fc1_weight, fc1_bias, fc2_weight, fc2_bias,,

        action = generate_rand_action(observation_state)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # target net does not update every single step, eval net does.
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        # choose a random record to store in target network
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_S_A])
        b_a = torch.LongTensor(b_memory[:, N_S_A:N_S_A*2].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_S_A*2:N_S_A*2 + 1])
        b_s_ = torch.FloatTensor(b_memory[:, -N_S_A:])
        # print(self.eval_net(b_s))
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        '''
        if self.Big_timer < 10:
            print("q_next: ", q_next.max(1)[0].view(BATCH_SIZE, 1))
            print("q_eval: ", q_eval)
            self.Big_timer += 1
        '''
        # torch.max[0] = max_value, torch.max[1] = index
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


'''
Read vec_list part
'''

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]
w_b_txt = sys.argv[2]
f = open(py_text, 'r')
wr = open(w_b_txt, 'w+')
lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

vec_num = [[0.0 for _ in range(625)] for _ in range(1000)]  # numbers of maps
# Read file
iter_time = -1
for line in lines:
    search_ = False
    map_pos = 0
    temp = ""
    for i in line:
        if i == "i":  # ignore iter line
            iter_time += 1
            break
        if search_:  # Starting to search everything
            if i == "]":  # End of the line
                temp = ""
                search_ = False
            elif i == " ":  # pri of one map is collected
                vec_num[iter_time][map_pos] = float(temp)
                map_pos += 1
                temp = ""
            else:  # collecting pri of map
                temp += i
        if i == "[":
            search_ = True

f.close()

'''
DQN part
'''

dqn = DQN()
env = environment(vec_num)
# MEMORY_CAPACITY time of collecting exp,
# epochs time of learning
# MEMORY_CAPACITY*epochs times total

print("Start exploration.")
for i_episode in range(epochs):
    env.reset()
    s = env.getStartstate()
    ep_r = 0
    while True:
        # Choose a random action on state s.
        a = dqn.choose_action(s)

        # take action
        # done: The end.
        # modify the reward
        s_, r, done = env.step(a)

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        # exploitation
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_


'''
print("fc1_weighted: \n", fc1_weight, "\n","fc2_bias: \n", fc1_bias, "\n","fc2_weighted: \n", fc2_weight, "\n",
      "fc2_bias: \n", fc2_bias, file=wr)
# 794*16, 16*1, 16*1, 1*1

for i in range(16):
    for j in range(794):
        print("start:", fc1_weight[i][j], end=" ")
    print("")
'''

torch.save(dqn.eval_net.state_dict(), 'eval_network.pt')
print('Save eval_network complete.')
torch.save(dqn.target_net.state_dict(), 'target_network.pt')
print('Save target_network complete.')
