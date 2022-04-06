import sys
import random
import numpy as np


'''

Problem: The random always pick the request in the front.
         Only save the small map

'''

# Parameters
B_size = 1000
map_num = 2161
iter_ = 200
Reward_fetch = -15.0


class Cache_state(object):
    def __init__(self):
        self.count = 0
        self.cache_state = [0] * B_size

    def Initial_cache(self):
        # Initialize cache_state
        for j in range(map_num):
            rand_cache = random.randint(0, 2)
            if self.count >= B_size:
                break
            elif rand_cache == 1:
                self.cache_state[self.count] = j
                self.count += 1

    def remove_and_lineup(self, j):  # j is the index of the Small_map_number
        main_change = self.cache_state[j]
        for b in reversed(range(j+1)):
            if b == 0:
                break
            else:
                self.cache_state[b] = self.cache_state[b-1]
        self.cache_state[0] = main_change

    def add_new(self, j):
        for b in reversed(range(B_size)):
            if b == 0:
                break
            else:
                self.cache_state[b] = self.cache_state[b-1]
        self.cache_state[0] = j


def getReward(cache_state, vec_num_one):
    # print("Cache: ", cache_state)
    # print("Vec: ", vec_num_one)
    reward_temp = 0
    temp_c = 0
    for j in range(map_num):  # From map j in time i
        if j in cache_state:  # Cached by Small m
            continue
        else:
            temp_c += 1
            reward_temp += Reward_fetch * vec_num_one[j]
    print("c_ : ", temp_c)

    return reward_temp


'''
Read vec_list part
'''

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]
f = open(py_text, 'r')
lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

vec_num = [[0.0 for _ in range(map_num)] for _ in range(iter_)]  # numbers of maps
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

c_ = Cache_state()
c_.Initial_cache()
ep_r = 0.0
for i in range(iter_):
    # Reward
    if i >= 1:
        r_ = getReward(c_.cache_state, vec_num[i])
        print(r_)
        ep_r += r_
    # change content
    for j in range(map_num):
        if vec_num[i][j] > 0 :
            if j in c_.cache_state:  # j is Small_map_number
                index_ = c_.cache_state.index(j)
                c_.remove_and_lineup(index_)
            else:
                c_.add_new(j)
    # print(c_.cache_state)

print("Total reward: ", ep_r)





