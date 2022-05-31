import sys
import torch
import torch.nn as nn
import numpy as np
import MIP_only_gurobi as gb
import logging
import os
import signal
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
MUST READ:
    The total map data is 200 iterations now.
'''

# Not important
timer_output_Action = 0
start = time.time()

iter_ = 10
Small_map_number = 1010
Big_map_number = 441
Total_map_number = 1451
B_size = 100
D_size = 15

# Hyper Parameters
# The parameter interfere the training time
BATCH_SIZE = 16
MEMORY_CAPACITY = 128
GAME_STEP_NUM = iter_ - 1  # (for the zero)
EPOCHS = 500
LR = 0.01  # learning rate
EPSILON = 0.8  # greedy policy
decay_rate = 0.8
GAMMA = 0.8  # reward discount
TARGET_REPLACE_ITER = 100 # target update frequency
# Network parameters
input_dim = Total_map_number  # Q() = S x A
hidden_dim = 16
output_dim = 1
Reward_hit = 10.0
Reward_fetch = 20.0
Game_step = 0
N_S_A = input_dim
s = [0 for i in range(Total_map_number)]


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    # create a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


# signal_catcher
def signal_handler(signum):
    print('signal_handler: caught signal ' + str(signum))
    if signum == signal.SIGINT.value:
        print('Force to stop ')

        sys.exit(1)

def get_state_action():
    print("get_state")
    gen_state = np.random.choice([0.0, 1.0], size=N_S_A, p=[0.65, 0.35])
    count = 0
    for i in gen_state:
        if i == 1:
            count += 1
    while count > B_size:  # Normally it will not need this.
        rand_pos = np.random.randint(0, N_S_A)
        if gen_state[rand_pos] == 1:
            gen_state[rand_pos] = 0
            count -= 1
        else:
            continue
    return gen_state


def generate_rand_action(s_in):
    print("Random action")
    action = s_in.copy()
    for i in range(D_size):
        if action[i] == 0:
            action[i] = 1
        elif action[i] == 1:
            action[i] = 0

    return action


class Environment(object):
    def __init__(self, vec_num_in):
        self.states = None
        self.vec_num = vec_num_in
        self.done = False

    def reset(self):
        self.states = None
        global Game_step
        Game_step = 0
        self.done = False

    def get_Start_state(self):
        self.states = get_state_action()
        return self.states

    def step(self, a_in):
        global Game_step
        self.states = a_in.copy()  # Since our action and state is the same format, I change them directly.
        #  self.states is the s_ now.
        r_ = self.getReward()
        if Game_step == GAME_STEP_NUM:
            self.done = True
        else:
            Game_step += 1
        return self.states, r_, self.done

    def getReward(self):
        reward_temp = 0
        global Game_step
        for j in range(Small_map_number):  # From map j in time i
            if self.states[j] == 1:  # Cached by Small m
                continue
            else:
                if self.states[Small_map_number + map_[j]] == 1:  # Cached by Big M
                    reward_temp += Reward_hit * self.vec_num[Game_step][j]
                else:  # Fail, and fetching from data set
                    reward_temp += Reward_fetch * self.vec_num[Game_step][j]

        return reward_temp


# @staticmethod
def choose_action(observation_state, vec_num_in):
    # input only one sample
    global Game_step
    gurobi_func = gb.Gurobi(vec_num_in[Game_step],
                            t_minus_s_=observation_state, B_size=B_size, D_size=D_size, Reward_hit=Reward_hit,
                            Reward_fetch=Reward_fetch, Total_map=Total_map_number, Small_number=Small_map_number
                            , hidden_dim=hidden_dim, map_=map_, GAMMA=GAMMA)
    action = gurobi_func.MIP_formulation()

    # if self.learn_step_counter % 10 == 0:
    '''
    print("vec_num:\n", vec_num_in[Game_step], file=py_out)
    
    print("Action \n", " :[ ", end="", file=py_out)
    for i in action:
        print(i, " ", end="", file=py_out)
    print("]\n", end="", file=py_out)
    '''
    return action

# Initial log
log_ = log_creater("log_file")

# GPU
if torch.cuda.is_available():
    cuda0 = torch.device('cuda:0')
    print(cuda0, "name: ", torch.cuda.get_device_name(0))

'''
Read vec_list part
'''

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]
py_map = sys.argv[2]

f = open(py_text, 'r')
ptr_in = open(py_map, "r")


'''
# Map formation #
'''
lines = (line.rstrip() for line in ptr_in.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

map_ = [0] * Small_map_number  # 0~674

for line in lines:
    # print(line)
    temp = line.split(' ')
    for i in temp:
        temp_in = i.split(',')
        try:
            map_[int(temp_in[0])] = int(temp_in[1])  # temp_in[0]:index, temp_in[1]: Big map index
            # print(int(temp_in[0]), int(temp_in[1]))
        except:
            print(int(temp_in[0]), "  ", int(temp_in[1]))
            sys.exit()
ptr_in.close()

'''
map_max = 0
# Get the big map max
for i in range(len(map_)):
    # print("i: {}, map_: {}".format(i, map_[i]))
    if map_[i] > map_max:
        map_max = map_[i]

print("map_max: ", map_max)
'''

'''
# vec_num #
'''
# f
lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

vec_num = [[0.0 for _ in range(Small_map_number)] for _ in range(iter_)]  # numbers of maps

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

# signal_catcher
signal.signal(signal.SIGINT, signal_handler)

env = Environment(vec_num)
# MEMORY_CAPACITY time of collecting exp,
# epochs time of learning
# MEMORY_CAPACITY*epochs times total

print("Start exploration.")
env.reset()
s = env.get_Start_state()

# s = np.zeros(N_S_A, dtype=float)
ep_r = 0

end_r = 0
rand_iter = 10
for i in range(rand_iter):
    env.reset()
    s = env.get_Start_state()
    ep_r = 0
    while True:
        # Choose a random action on state s.
        a = choose_action(s, vec_num)  # time_round += 1
        # take action
        # done: The end.
        # modify the reward
        s_, r, done = env.step(a)
        # ep_r: total reward
        ep_r += r

        if done:
            log_.info('Ep: %s'  '| Ep_r: %f', i, round(ep_r, 2))
            break

        s = s_
    end_r += ep_r

temp_ = end_r/rand_iter
log_.info('Average Ep_r: %f', 0, round(temp_, 2))


end = time.time()

log_.info("Total used time: {}".format(end - start))
