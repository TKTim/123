from gurobipy import *


class Gurobi:
    def __init__(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias, vec_num, t_minus_s_, Reward_hit, Reward_fetch,
                 D_size, B_size, Total_map, Small_number, hidden_dim, map_):
        self.f1_w = fc1_weight
        self.f1_b = fc1_bias
        self.f2_w = fc2_weight
        self.f2_b = fc2_bias
        self.vec_num = vec_num
        self.t_minus_s_ = t_minus_s_.copy()
        self.M_plus = 1000
        self.M_minus = -1000
        self.Total_map = Total_map
        self.Small_number = Small_number
        self.D_size = D_size
        self.B_size = B_size
        self.hidden_dim = hidden_dim
        self.map_ = map_
        self.Reward_hit_positive = Reward_hit
        self.Reward_fetch_positive = Reward_fetch
        # self.Reward_hit_positive = Reward_hit
        # self.Reward_fetch_positive = Reward_fetch
        # self.action = [0 for _ in range(425)]
        self.M = 0.5
        self.time_round = 0


    def MIP_formulation(self):
        # Create a new model
        m = Model("mip1")
        m.params.NonConvex = 2

        # Create variables
        s_ = m.addVars(range(self.Total_map), vtype=GRB.BINARY, ub=1, lb=0,  name='s_')
        # x_ = m.addVars(range(self.Total_map), ub=1, lb=0,  name='re_s_')
        hit = m.addVars(range(self.Small_number), vtype=GRB.BINARY, ub=1, lb=0,  name='hit_')
        fetch = m.addVars(range(self.Small_number), vtype=GRB.BINARY, ub=1, lb=0,  name='fetch_')
        d_m_ = m.addVars(range(self.Total_map), vtype=GRB.BINARY, ub=1, lb=0,  name='d_m_')
        y_h = m.addVars(range(self.hidden_dim), lb=0, name="y_h")
        z_h = m.addVars(range(self.hidden_dim), vtype=GRB.BINARY, ub=1, lb=0, name="z_h")
        #V_ = m.addVar(name="V_")

        # Integrate new variables
        m.update()

        # Set objective
        m.setObjective(quicksum(self.vec_num[i] * hit[i] * self.Reward_hit_positive for i in range(self.Small_number)) +
                       quicksum(self.vec_num[i] * fetch[i] * self.Reward_fetch_positive for i in range(self.Small_number)) +
                       quicksum(y_h[i] * self.f2_w[i] for i in range(self.hidden_dim)) + self.f2_b, GRB.MINIMIZE)

        # Set constraints
        # cons. 1~4

        for i in range(self.hidden_dim):  # self.hidden_dim:H
            try:
                # 求出y_h max， f1_w 正負，x_設 {0, 1}
                m.addConstr(y_h[i] >= quicksum(self.f1_w[i][j] * (1 - s_[j]) for j in range(self.Total_map)) + self.f1_b[i], "c1_")
                # self.M_minus =

                m.addConstr(
                    y_h[i] <= quicksum(self.f1_w[i][j] * (1 - s_[j]) for j in range(self.Total_map)) + self.f1_b[i] - self.M_minus * (
                            1 - z_h[i]), "c2_")
                m.addConstr(y_h[i] <= self.M_plus * z_h[i], "c3_")
                # m.addConstr(y_h[i] >= 0)

            except:
                print("Error")

        # cons. x_
        '''
        for i in range(self.Total_map):
            try:
                m.addConstr(x_[i] == 1 - s_[i], "cx_")
            except:
                print("Fail")
        '''

        # cons. Miss,hit  Miss,fetch
        for i in range(self.Small_number):
            try:
                m.addConstr(hit[i] == (1 - s_[i]) * s_[self.Small_number + self.map_[i]], "hit_")  # (m is not) and (M is) <hit>
                # m.addConstr(hit[i] <= 1-s_[i], "hit_s")
                # m.addConstr(hit[i] >= s_[self.Small_number + self.map_[i]] - s_[i], "hit_b")

                m.addConstr(fetch[i] == (1 - s_[i]) * (1 - s_[self.Small_number + self.map_[i]]), "fetch_")
                # m.addConstr(fetch[i] >= 1 - s_[i] - s_[self.Small_number + self.map_[i]], "fetch_")
            except:
                print("Fail H F.")
                sys.exit()

        # Download parameters

        # cons. 5~7
        for i in range(self.Total_map):
            try:
                m.addConstr(d_m_[i] >= s_[i] - self.t_minus_s_[i])

            except:
                print("Fail d_m")
                break

        # cons. 7~8
        try:
            m.addConstr(quicksum(d_m_[i] for i in range(self.Total_map)) <= self.D_size)  # Dowload size
            m.addConstr(quicksum(s_[i] for i in range(self.Total_map)) <= self.B_size, "c_size")
        except:
            print("Fail sum")

        # m.feasRelaxS(0, True, False, True)
        m.setParam('TimeLimit', 60)

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

        # quicksum(y_h[i] * self.f2_w[i] for i in range(self.hidden_dim)) + self.f2_b
        y_temp = []
        len_ = 0
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))
            if v.varName[0] == "y":
                y_temp.append(v.x)
            if len_ < self.Total_map:
                try:
                    self.t_minus_s_[len_] = int(v.x)
                except:
                    print(int(v.x))
                    print(len_)
                    print("TM: ", self.Total_map)
                    print(len(self.t_minus_s_))
                    sys.exit()
                len_ += 1
        print('Obj: %g' % m.objVal)

        # V_
        V_ = 0
        for i in range(len(y_temp)):
            V_ += y_temp[i] * self.f2_w[i]
        V_ += self.f2_b

        return self.t_minus_s_, V_  # The update of t_minus_x is the same meaning of action
