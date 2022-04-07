import sys
import torch
import torch.nn as nn
import numpy as np
import gurobi_sav_lp as gb
import logging
import os
import signal
import time
import random
import torch.backends.cudnn
# lock on seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
MUST READ:
    The total map data is 200 iterations now.
'''

# Not important
timer_output_Action = 0
start = time.time()

# Map
iter_ = 10
Small_map_number = 376
Big_map_number = 298
Total_map_number = 674
B_size = 15
D_size = 3

# Hyper Parameters
# The parameter interfere the training time
BATCH_SIZE = 16
MEMORY_CAPACITY = 128
GAME_STEP_NUM = iter_ - 2  # (for the zero)
EPOCHS = 1000
LR = 0.01  # learning rate
EPSILON = 0.8  # greedy policy
decay_rate = 0.8
GAMMA = 0.8  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
# Network parameters
input_dim = Total_map_number*2 + Small_map_number  # Q() = S{total} x V{small}
hidden_dim = 16
output_dim = 1
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


def get_Weight_and_Bias(model):
    num = 0
    for par in model.parameters():
        if num == 0:
            fc1_weight = par.cpu().detach().numpy().tolist()
        elif num == 1:
            fc1_bias = par.cpu().detach().numpy().tolist()
        elif num == 2:
            fc2_weight_temp = par.cpu().detach().numpy().tolist()
        else:
            fc2_bias_temp = par.cpu().detach().numpy().tolist()
        num += 1

    # 0: fc1.weight: torch.Size([16, 794])
    # 1: fc1.bias: torch.Size([16])
    # 2: fc2.weight: torch.Size([1, 16])
    # 3: fc2.bias: torch.Size([1])

    fc2_weight = [0.0 for _ in range(hidden_dim)]
    for i in fc2_weight_temp:
        for j in range(hidden_dim):
            fc2_weight[j] = i[j]
    fc2_bias = fc2_bias_temp[0]

    # print("fc1_weight:", fc1_weight, file=py_out)
    # print("fc1_bias:", fc1_bias, file=py_out)
    # print("fc2_weight:\n", fc2_weight, file=py_out)
    # print("fc2_bias:\n", fc2_bias, file=py_out)

    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


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
    count = 0
    action = s_in.copy()
    for i in action:
        if i == 1:
            count += 1
    i = 0
    while i < D_size:
        rand_pos = np.random.randint(0, N_S_A)
        if action[rand_pos] == 0 and count < B_size:
            action[rand_pos] = 1
            i += 1
            count += 1
        elif action[rand_pos] == 1 and count == B_size:  # is full
            action[rand_pos] = 0
            rand_pos = np.random.randint(0, N_S_A)
            while action[rand_pos] == 1:
                rand_pos = np.random.randint(0, N_S_A)
            action[rand_pos] = 1
            i += 1

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
        # self.states = get_state_action()
        self.states = [0] * Total_map_number
        for i in range(B_size):
            self.states[i] = 1
        # print("self.states: ", self.states)
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
            if self.states[j] == 1:  # Cached by Small m
                continue
            else:
                if self.states[Small_map_number + map_[j]] == 1:  # Cached by Big M
                    reward_temp += Reward_hit * self.vec_num[Game_step][j]
                else:  # Fail, and fetching from data set
                    reward_temp += Reward_fetch * self.vec_num[Game_step][j]

        return reward_temp


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim_in):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim_in
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_dim)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.to(device=cuda0)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(input_dim, hidden_dim, output_dim), Net(input_dim, hidden_dim, output_dim)
        try:
            self.eval_net.load_state_dict(torch.load('./saved/eval_network.pt'))
            self.target_net.load_state_dict(torch.load('./saved/target_network.pt'))
            log_.info("Success load previous dict.")
        except FileNotFoundError:
            print("No saved files.")
        self.Big_timer = 0
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = torch.zeros(( MEMORY_CAPACITY, N_S_A * 3 + 2)).to(device=cuda0)  # initialize memory ( S + A + 'r' + S_ + 'g_step' + V )
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.timer_output_Action = 0

    # Change to ILP Solver

    # @staticmethod
    def choose_action(self, observation_state):
        # input only one sample
        if np.random.uniform() < EPSILON:  # Gurobi Solver
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = get_Weight_and_Bias(dqn.eval_net)
            global Game_step

            # print fc...
            '''
            for i in fc1_weight:
                print("fc1_w ", i)
            for i in fc1_bias:
                print("fc1_b", i)
            for i in fc2_weight:
                print("fc2_w", i)
            print("fc2_b", fc2_bias)
            '''
            gurobi_func = gb.Gurobi(fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num[Game_step],
                                    t_minus_s_=observation_state, B_size=B_size, D_size=D_size, Reward_hit=Reward_hit,
                                    Reward_fetch=Reward_fetch, Total_map=Total_map_number, Small_number=Small_map_number
                                    , hidden_dim=hidden_dim, map_=map_, GAMMA=GAMMA)
            action, V_ = gurobi_func.MIP_formulation()
            '''
            print("Action \n", " :[ ", end="", file=py_out)
            for i in action:
                print(i, " ", end="", file=py_out)
            print("]\n", end="", file=py_out)
            '''
            '''
            x_ = action.copy()
            for j in range(len(x_)):
                if x_[j] <= 0.5:
                    x_[j] = 1.0
                else:
                    x_[j] = 0.0
            x_ = torch.tensor(x_, dtype=torch.float32)
            print("x_:  ", x_)
            vec_in = torch.tensor(vec_num[Game_step + 1], dtype=torch.float32)
            print("vec_in:  ", vec_in)
            In_q_eval = torch.cat((x_, vec_in), 0).to(device=cuda0)

            test_eval = self.eval_net(In_q_eval)

            print("V_:  ", V_, file=py_out)
            print("test_eval_:  ", test_eval, file=py_out)
            print("In_q_eval:  ", In_q_eval, file=py_out)
            '''

        else:  # random
            action = generate_rand_action(observation_state)
            # print("Random action", file=py_out)

        return action, V_

    def store_transition(self, s_in, a_in, r_in, s__in, g_step):

        transition = np.hstack((s_in, a_in, r_in, s__in, g_step))
        transition = torch.tensor(transition)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        global Game_step
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # target net does not update every single step, eval net does.
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        # choose a random record to store in target network
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        b_memory = self.memory[sample_index, :].to(device=cuda0)
        b_s = b_memory[:, :N_S_A].to(device=cuda0)
        b_a = b_memory[:, N_S_A:N_S_A * 2].to(device=cuda0)
        b_r = b_memory[:, N_S_A * 2:N_S_A * 2 + 1].to(device=cuda0)
        b_s_ = b_memory[:, N_S_A * 2 + 1:N_S_A * 3 + 1].to(device=cuda0)  # batch * |b_s_|
        g_step = b_memory[:, N_S_A * 3 + 1:N_S_A * 3 + 2].to(device=cuda0)  # V_t
        # print("g_step:\n", g_step, file=py_out)

        # print("b_s:\n", b_s, file=py_out)
        # print("b_a:\n", b_a, file=py_out)
        # print("b_r:\n", b_r, file=py_out)
        # print("b_s_:\n", b_s_, file=py_out)

        '''
        q_eval = self.eval_net(b_s)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagation
        '''
        # Data preprocessing
        x_ = b_s_.clone()  # A bar
        for i in range(len(x_)):
            for j in range(N_S_A):
                if x_[i][j] <= 0.5:
                    x_[i][j] = 1.0
                else:
                    x_[i][j] = 0.0

        y_ = b_s.clone()  # S

        vec_batch_ = []
        for i in range(BATCH_SIZE):
            vec_batch_.append(vec_num[int(g_step[i])])  # V_t

        vec_batch_ = torch.tensor(vec_batch_, dtype=torch.float32).to(device=cuda0)

        In_q_eval = torch.cat((x_, vec_batch_, y_), 1)  # A bar, V, S bar

        # Working with network
        q_eval = self.eval_net(In_q_eval).to(device=cuda0)  # 16
        # q_next = self.target_net(b_s_).detach().to(device=cuda0)  # float.Tensor
        q_next = GetTargetS_(b_s_, dqn.target_net, g_step).to(device=cuda0)  # s' ,a' ,V'

        q_target = b_r + GAMMA * q_next
        # print("b_r: ", b_r, file=py_out)
        # print("q_eval  ", q_eval, file=py_out)
        # print("q_next  ", q_next, file=py_out)
        # print("q_target  ", q_target, file=py_out)
        loss = self.loss_func(q_eval, q_target).to(device=cuda0)
        # print("loss  ", loss, file=py_out)

        # loss = Variable(loss, requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


'''
def GetEvalS_(s_, eval_net, vec_num_in, g_step):
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = get_Weight_and_Bias(eval_net)
    V_s = []
    for i in range(BATCH_SIZE):
        gurobi_func = gb.Gurobi(fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num_in[int(g_step[i])],
                                t_minus_s_=s_[i], B_size=B_size, D_size=D_size, Reward_hit=Reward_hit,
                                Reward_fetch=Reward_fetch, Total_map=Total_map_number, Small_number=Small_map_number
                                , map_=map_)

        action, V_s_single = gurobi_func.MIP_formulation()
        V_s.append(V_s_single)

    V_s = torch.FloatTensor(V_s)

    return V_s
'''


# Get the \hatV value from Target_network
def GetTargetS_(s_new, target_net, g_step):
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = get_Weight_and_Bias(target_net)
    s_new_arr = s_new.clone()
    s_new_arr = s_new_arr.cpu().detach().numpy()
    V_s_ = []
    for i in range(BATCH_SIZE):
        g_s = g_step[i]
        if g_s == GAME_STEP_NUM:
            V_s_new_single = 0.0
        else:
            gurobi_func = gb.Gurobi(fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num=vec_num[int(g_s)+1],
                                    t_minus_s_=s_new_arr[i], B_size=B_size, D_size=D_size, Reward_hit=Reward_hit,
                                    Reward_fetch=Reward_fetch, Total_map=Total_map_number, Small_number=Small_map_number
                                    , hidden_dim=hidden_dim, map_=map_, GAMMA=GAMMA)
            action, V_s_new_single = gurobi_func.MIP_formulation()

            '''
            x_ = action
            for i in range(len(action)):
                x_[i] = 1 - x_[i]

            # 0: fc1.weight: torch.Size([16, 794])
            # 1: fc1.bias: torch.Size([16])
            # 2: fc2.weight: torch.Size([1, 16])
            # 3: fc2.bias: torch.Size([1])
            h_ = [0.0] * hidden_dim
            for i in range(hidden_dim):  # h
                for j in range(input_dim):
                    h_[i] += x_[j] * fc1_weight[i][j]
                h_[i] += fc1_bias[i]
            end_ans = 0.0

            for i in range(len(h_)):
                if h_[i] < 0:
                    h_[i] = 0

            for i in range(hidden_dim):
                end_ans += h_[i] * fc2_weight[i]
            end_ans += fc2_bias

            q_eval = target_net.forward(x_)
            print("Q_:  ", q_eval, file=py_out)
            print("hatV_:  ", V_s_new_single, file=py_out)
            print("end_ans: ", end_ans, file=py_out)
            '''
            # V_s_new_single

        V_s_.append([V_s_new_single])

    V_s_ = torch.FloatTensor(V_s_)
    return V_s_


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

dqn = DQN()
env = Environment(vec_num)
# MEMORY_CAPACITY time of collecting exp,
# epochs time of learning
# MEMORY_CAPACITY*epochs times total
num = 0
print("Start exploration.")
for i_episode in range(EPOCHS):
    env.reset()
    s = env.get_Start_state()
    # print("state: ", s)
    ep_r = 0
    while True:
        # print(s)
        # Choose a random action on state s.
        # print("Game_step: ", Game_step, file=py_out)
        a, V_ = dqn.choose_action(s)  # time_round += 1
        if V_ == -1:

        # take action
        # done: The end.
        # modify the reward
        # store v[game_step]

        g_s = Game_step
        s_, r, done = env.step(a)  # have to after anything about Game_step
        # print("r: ", r, file=py_out)
        dqn.store_transition(s, a, r, s_, g_s)

        # ep_r: total reward
        ep_r += r
        # exploitation
        if dqn.memory_counter > MEMORY_CAPACITY:
            loss = dqn.learn()
            log_.info('Ep: %s'  '| Ep_loss: %f', i_episode, loss)
            if done:
                log_.info('Ep: %s'  '| Ep_r: %f', i_episode, round(ep_r, 2))
                if i_episode % 3 == 0 and i_episode != 0:
                    torch.save(dqn.eval_net.state_dict(), 'saved/eval_network_' + str(num) + '.pt')
                    # print('Save eval_network complete.')
                    torch.save(dqn.target_net.state_dict(), 'saved/target_network_' + str(num) + '.pt')
                    # print('Save target_network complete.')
                    log_.info("Saved succeed", i_episode)
                    num += 1
        if done:
            break
        s = s_

    # EPSILON Decay
    if EPSILON < 1.0:
        if i_episode % 10:
            EPSILON = 1 - (1 - EPSILON) * decay_rate

end = time.time()
log_.info("Total used time: {}".format(end - start))
