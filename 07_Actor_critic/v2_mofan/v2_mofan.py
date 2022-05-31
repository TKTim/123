import os
import time
import tensorflow.compat.v1 as tf
import numpy as np
import sys
import logging
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
MUST READ:
    The total map data is 200 iterations now.
'''


#####################  hyper parameters  ####################

# Not important
timer_output_Action = 0
start = time.time()

# Map
iter_ = 10
Small_map_number = 1010
Big_map_number = 441
Total_map_number = 1451
B_size = 100
D_size = 15

# Hyper Parameters
# The parameter interfere the training time
MEMORY_CAPACITY = 3000
BATCH_SIZE = 64
GAME_STEP_NUM = iter_ - 2  # (for the zero)
EPOCHS = 5000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
EPSILON = 0.8  # greedy policy
decay_rate = 0.8
GAMMA = 0.8     # reward discount
TAU = 0.01      # soft replacement
TARGET_REPLACE_ITER = 100  # target update frequency
# Network parameters
Reward_fetch = -20.0
Game_step = 0
N_S_A = Total_map_number
s = [0 for i in range(Total_map_number)]


###############################  functions  ###############################

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

def generate_rand_action(s_in):
    print("Random action")
    action = [0] * B_size
    i = 0
    while i < D_size:
        rand_ = np.random.randint(0, B_size)
        if action[rand_] != 1:
            action[rand_] = 1
            i += 1
        else:
            continue

    return action


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim, 
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a_out = [0] * B_size

        with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
        

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        if np.random.uniform() < EPSILON:
            # a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
            s_ = s.copy()
            s_ = np.array(s_)
            s_ = s_[np.newaxis, :]
            # print("size of s_: ", len(s_[0]))
            a = self.sess.run(self.a, {self.S: s_})
            print("choose_action: \n", a )
            count = 0
            threshold = 0.9
            i = 0
            # a_sorted = sorted(a[0], reverse=True)
            while count < B_size:  # loop
                if i == B_size:
                    threshold -= 0.1
                    # print("threshold: ", threshold)
                    i = 0
                if count >= D_size:
                    break
                if a[0][i] >= threshold:
                    self.a_out[i] = 1
                    count += 1
                i += 1
            
            return self.a_out
            
            
        else:
            a = generate_rand_action(s)
            print("generate_rand_action....")
            
            return a

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = s + a + [r] + s_
        print("s:\n",s)
        print("a:\n",a)
        print("r: \n", r)
        # print("s':\n",s_)
        # print("transition: \n",)
        # print("size of trans: ", len(transition))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.tanh, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 30, activation=tf.nn.tanh, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            '''
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            '''
            sa = tf.concat((s, a), 1)
            sa = tf.layers.dense(sa, 512,activation=tf.nn.relu, name='lc_1', trainable=trainable)
            sa = tf.layers.dense(sa, 300,activation=tf.nn.relu, name='lc_2', trainable=trainable)
            return tf.layers.dense(sa, 1, trainable=trainable)  # Q(s,a)
        
        
###############################  Environment  ####################################
        
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
        # self.states = get_state_action()
        self.states = [0] * B_size
        for i in range(B_size):
            self.states[i] = i
        # print("self.states: ", self.states)
        return self.states

    def step(self, a_in):
        global Game_step
        
        candidate_map = [[0.0 for _ in range(2)] for _ in range(B_size)]
        # print("can: \n", candidate_map)
        pos_candidate = 0
        for i in range(Small_map_number):
            if vec_num[Game_step][i] > 0.0 and i not in self.states:  # priority > 0.0, not in states already
                if pos_candidate >= B_size:
                    break
                candidate_map[pos_candidate][0] = i  # index
                candidate_map[pos_candidate][1] = vec_num[Game_step][i]  # priority
                pos_candidate += 1
        candidate_map.sort(key = lambda s: s[1])
        
        #self.states = a_in.copy()  # Since our action and state is the same format, I change them directly.
        pos_candidate = 0
        for i in range(B_size):
            if a_in[i] == 1.0:
                self.states[i] = candidate_map[pos_candidate][0]
                pos_candidate += 1
                # unfinished
             
        #  self.states is the s_ now.
        r_ = self.getReward()
        if Game_step == GAME_STEP_NUM:
            self.done = True
        else:
            Game_step += 1
            
        return self.states, r_, self.done

    def getReward(self):
        reward_temp = 0
        for j in range(Small_map_number):  # go all vec_num
            if vec_num[Game_step][j] > 0.0 and j not in self.states:
                reward_temp += vec_num[Game_step][j] * Reward_fetch

        return reward_temp



###############################  Read file  ###################################

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


log_ = log_creater("log_file")


###############################  training  ####################################
print("Starting main ...")
env = Environment(vec_num)

s_dim = B_size + Small_map_number  # cache x Vec
a_dim = B_size

ddpg = DDPG(a_dim, s_dim)

var = 3  # control exploration
t1 = time.time()
for i_episode in range(EPOCHS):
    env.reset()
    s = env.get_Start_state()
    ep_reward = 0
    while True:
        s_scale = s.copy()
        for i in range(len(s_scale)):
            s_scale[i] = s_scale[i]/Small_map_number
        s_in = s_scale + vec_num[Game_step]  # Network states = cache status x V
        # print("States: \n", s)
        a = ddpg.choose_action(s_in)  # choose action
        # print("Actions: \n", a)
        # print("Actions_size: \n", len(a))
        # a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done = env.step(a)
        # print("State prime: \n", s_)
        s_scale_ = s_.copy()
        # Scale states
        for i in range(len(s_scale)):
            s_scale_[i] = s_scale_[i]/Small_map_number
            
        s_in_ = s_scale_ + vec_num[Game_step]  # Network states = cache status x V
        # ddpg.store_transition(s, a, r / 10, s_)
        ddpg.store_transition(s_in, a, r, s_in_)
        ep_reward += r
        if ddpg.pointer > MEMORY_CAPACITY:
            print("Start Learning...")
            var *= .9995    # decay the action randomness
            ddpg.learn()
            if done:
                log_.info('Ep: %s'  '| Ep_r: %f', i_episode, round(ep_reward, 2))

        s = s_

        if done:
            # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
        
print('Running time: ', time.time() - t1)