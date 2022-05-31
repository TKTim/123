import sys
import numpy as np
import logging
import os
import signal
import time
import random

seed = 123
np.random.seed(seed)
random.seed(seed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Not important
timer_output_Action = 0
start = time.time()

random_iter = 10

# Map
iter_ = 10
GAME_STEP_NUM = iter_-1
Small_map_number = 1010
Big_map_number = 441
Total_map_number = 1451
B_size = 100
D_size = 15
Reward_hit = 10.0
Reward_fetch = 20.0
Game_step = 0
N_S_A = Total_map_number
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
    # print("get_state")
    # gen_state = np.zeros(B_size)
    gen_state = random.sample(range(0, Small_map_number), B_size)
    print("State: \n",gen_state)

    return gen_state


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
        for j in range(Small_map_number):  # From map j in time i
            if j in self.states:
                continue
            else:
                reward_temp += Reward_fetch * self.vec_num[Game_step][j]

        return reward_temp


def choose_action(observation_state):
    index_zero = np.arange(0, Small_map_number, 1)
    formal_list = [[0 for _ in range(2)] for _ in range(Small_map_number)]
    for i in range(Small_map_number):
        formal_list[i][0] = index_zero[i]
        formal_list[i][1] = vec_num[Game_step][i]

    a_in = sorted(formal_list, key=lambda  s: s[1])
    a_out = np.zeros(B_size)
    cache = 0
    download = 0
    for i in range(Small_map_number):
        if cache >= B_size:
            break
        if a_in[i][0] in observation_state:
            a_out[cache] = a_in[i][0]
            cache += 1
        elif download <= D_size:
            a_out[cache] = a_in[i][0]
            cache += 1
            download += 1

    return a_out

# Initial log
log_ = log_creater("log_file")


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
                try:
                    vec_num[iter_time][map_pos] = float(temp)
                except:
                    print("iter_time, map_pos  ", iter_time, map_pos)
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
num = 0
print("Start exploration.")
for i_episode in range(random_iter):
    env.reset()
    s = env.get_Start_state()
    # print("state: ", s)
    ep_r = 0
    while True:
        a = choose_action(s)  # time_round += 1
        s_, r, done = env.step(a)  # have to after anything about Game_step
        ep_r += r

        if done:
            log_.info('Ep: %s'  '| Ep_r: %f', i_episode, round(ep_r, 2))
            break
        s = s_


end = time.time()
log_.info("Total used time: {}".format(end - start))
