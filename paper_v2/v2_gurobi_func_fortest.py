import random
import sys

from gurobipy import *

'''
Read vec_list part
'''


def getBigmap(i):

    loc_temp = int(i / 50)
    line_temp = int(i / 100)
    loc = int(i - 50 * loc_temp)
    map_loc = int(loc / 2)
    ans = int(map_loc + 25 * line_temp)

    return ans


class Gurobi:
    def __init__(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num, t_minus_s_):
        self.f1_w = fc1_weight
        self.f1_b = fc1_bias
        self.f2_w = fc2_weight
        self.f2_b = fc2_bias
        self.vec_num = vec_num
        self.t_minus_s_ = t_minus_s_
        self.M_plus = 1000
        self.M_minus = -1000
        self.D_size = 20
        self.B_size = 50
        # self.action = [0 for _ in range(425)]
        self.M = 0.5
        self.time_round = 0

    def MIP_formulation(self):
        # Create a new model
        m = Model("mip1")

        # Create variables
        s_ = m.addVars(range(425), ub=1, lb=0, vtype=GRB.BINARY, name='s_')
        x_ = m.addVars(range(425), ub=1, lb=0, vtype=GRB.BINARY, name='re_s_')
        hit = m.addVars(range(333), ub=1, lb=0, vtype=GRB.BINARY, name='hit_')
        fetch = m.addVars(range(333), ub=1, lb=0, vtype=GRB.BINARY, name='fetch_')
        d_m_ = m.addVars(range(425), ub=1, lb=0, vtype=GRB.BINARY, name='d_m_')
        y_h = m.addVars(range(16), name="y_h")
        z_h = m.addVars(range(16), vtype=GRB.BINARY, ub=1, lb=0, name="z_h")

        # Integrate new variables
        m.update()

        # Set objective
        m.setObjective(quicksum(self.vec_num[self.time_round][i] * hit[i] * 10 for i in range(333)) +
                       quicksum(self.vec_num[self.time_round][i] * fetch[i] * 12 for i in range(333)) +
                       quicksum(y_h[i] * self.f2_w[i] for i in range(16)) + self.f2_b
                       , GRB.MINIMIZE)

        # Set constraints
        # cons. 1~4

        for i in range(16):  # 16:H
            try:
                # 求出y_h max， f1_w 正負，x_設 {0, 1}
                m.addConstr(y_h[i] >= quicksum(self.f1_w[i][j] * x_[j] for j in range(425)) + self.f1_b[i])
                # self.M_minus =

                m.addConstr(
                    y_h[i] <= quicksum(self.f1_w[i][j] * x_[j] for j in range(425)) + self.f1_b[i] - self.M_minus * (
                            1 - z_h[i]))
                m.addConstr(y_h[i] <= self.M_plus * z_h[i])
                # m.addConstr(y_h[i] >= 0)

            except:
                print("Error")

        # cons. x_

        for i in range(425):
            try:
                # m.addConstr((s_[i] == 0) >> (x_[i] == 1))
                # m.addConstr((s_[i] == 1) >> (x_[i] == 0))
                m.addConstr(x_[i] == 1 - s_[i])
            except:
                print("Fail")


        # cons. Miss,hit  Miss,fetch
        for i in range(333):
            try:
                m.addConstr(hit[i] == and_(x_[i], s_[333 + getBigmap(i)]), "hit_")  # (m is not) and (M is) <hit>
                m.addConstr(fetch[i] == and_(x_[i], x_[333 + getBigmap(i)]), "fetch_")
            except:
                print("Fail H F.")
                sys.exit()

        # cons. 5~7
        for i in range(425):
            try:
                # Already try indicator (>>)
                m.addConstr(d_m_[i] >= s_[i] - self.t_minus_s_[i])
                # m.addConstr(d_m_[333+getBigmap(i)] == s_[333+getBigmap(i)] - self.t_minus_x_[333+getBigmap(i)])
                # m.addConstr(d_m_[i] >= 0)
                # m.addConstr(d_m_[333+getBigmap(i)] >= 0)

            except:
                print("Fail d_m")
                break


        # cons. 7~8
        try:
            m.addConstr(quicksum(d_m_[i] for i in range(425)) <= self.D_size)
            m.addConstr(quicksum(s_[i] for i in range(425)) <= self.B_size)
        except:
            print("Fail sum")


        # m.feasRelaxS(0, True, False, True)

        m.optimize()
        # print(m.display())
        # m.write('mip1.lp')

        # Infeasible test
        '''
        if m.status == GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % m.status)
            # do IIS, find infeasible constraints
            m.computeIIS()
            for c in m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
        '''

        len = 0
        for v in m.getVars():
            if len < 425:
                self.t_minus_s_[len] = int(v.x)
                len += 1
            print('%s %g' % (v.varName, v.x))
        print('Obj: %g' % m.objVal)

        return self.t_minus_s_  # The update of t_minus_x is the same meaning of action


if len(sys.argv) < 2:
    print('no argument')
    sys.exit()
py_text = sys.argv[1]
f = open(py_text, 'r')
lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
lines = (line for line in lines if line)  # Non-blank lines

# Non-Gurobi variables
t_minus_s_ = [0 for _ in range(425)]
# x = random.choice([0, 1], p=[0.5, 0.5], size=(100))
random.seed(10)
constrant_maps = 200
for j in range(425):
    rr = random.randint(0, 2)
    if constrant_maps <= 0:
        t_minus_s_[j] = 0  # Have changed
    else:
        if rr == 1:
            t_minus_s_[j] = 1
            constrant_maps -= 1
        else:
            t_minus_s_[j] = 0  # Have changed

U_ = 100
L_ = -100
vec_num = [[1.0 for _ in range(333)] for _ in range(1000)]  # numbers of maps

# W and B Random for test

fc1_weight = [[random.random() for _ in range(425)] for _ in range(16)]
fc1_bias = [random.random() for _ in range(16)]
fc2_weight = [random.random() for _ in range(16)]
fc2_bias = random.random()


'''
fc1_weight = [[0 for _ in range(425)] for _ in range(16)]
fc1_bias = [0 for _ in range(16)]
fc2_weight = [0 for _ in range(16)]
fc2_bias = 0
'''

# Read file
iter_time = -1
for line in lines:
    search_ = False
    map_pos = 0
    hit = ""
    for i in line:
        if i == "i":  # ignore iter line
            iter_time += 1
            break
        if search_:  # Starting to search everything
            if i == "]":  # End of the line
                hit = ""
                search_ = False
            elif i == " ":  # pri of one map is collected
                vec_num[iter_time][map_pos] = float(hit)
                map_pos += 1
                hit = ""
            else:  # collecting pri of map
                hit += i
        if i == "[":
            search_ = True

f.close()


gg = Gurobi(fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num, t_minus_s_)
for i in range(50):
    action = gg.MIP_formulation()
    gg.time_round += 1

print("Finished")
