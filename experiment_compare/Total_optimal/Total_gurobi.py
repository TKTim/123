from gurobipy import *


class Gurobi:
    def __init__(self, vec_num, t_minus_s_, Reward_hit, Reward_fetch,
                 D_size, B_size, Total_map, Small_number, hidden_dim, map_, GAMMA):
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
        self.GAMMA = GAMMA
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
        m.setParam('MIPGap', 1e-6)
        m.params.NonConvex = 2
        iter_ = 10
        # Create variables
        for i in range(0, iter_):
            # globals()['name'+str(i)]
            globals()['s_' + str(i)] = m.addVars(range(self.Total_map), vtype=GRB.BINARY, ub=1, lb=0,  name='s_' + str(i))
            globals()['hit_' + str(i)] = m.addVars(range(self.Small_number), vtype=GRB.BINARY, ub=1, lb=0,  name='hit_' + str(i))
            globals()['fetch_' + str(i)] = m.addVars(range(self.Small_number), vtype=GRB.BINARY, ub=1, lb=0,  name='fetch_' + str(i))
            globals()['d_m_' + str(i)] = m.addVars(range(self.Total_map), vtype=GRB.BINARY, ub=1, lb=0,  name='d_m_' + str(i))

        # Integrate new variables
        m.update()
        # print(m.display())

        # Set objective
        m.setObjective(
            quicksum(quicksum(self.vec_num[j][i] * globals()['hit_' + str(j)][i] * self.Reward_hit_positive for i in
                              range(self.Small_number))for j in range(iter_)) +
            quicksum(quicksum(self.vec_num[j][i] * globals()['fetch_' + str(j)][i] * self.Reward_fetch_positive for i in
                              range(self.Small_number)) for j in range(iter_)), GRB.MINIMIZE)

        # Set constraints
        # cons. Miss,hit  Miss,fetch
        for j in range(iter_):
            for i in range(self.Small_number):
                try:
                    m.addConstr(globals()['hit_' + str(j)][i] <= 1-globals()['s_' + str(j)][i], "hit_s")
                    m.addConstr(globals()['hit_' + str(j)][i] >= globals()['s_' + str(j)][self.Small_number + self.map_[i]] - globals()['s_' + str(j)][i], "hit_b")

                    m.addConstr(globals()['fetch_' + str(j)][i] >= 1 - globals()['s_' + str(j)][i] -
                                globals()['s_' + str(j)][self.Small_number + self.map_[i]], "fetch_")
                except:
                    print("Fail H F.")
                    sys.exit()

        # Download parameters

        # cons. 5~7
        for j in range(iter_):
            if j == 0:
                for i in range(self.Total_map):
                    m.addConstr(globals()['d_m_' + str(j)][i] >= globals()['s_' + str(j)][i] - self.t_minus_s_[i])
            else:
                for i in range(self.Total_map):
                    try:
                        m.addConstr(globals()['d_m_' + str(j)][i] >= globals()['s_' + str(j)][i] - globals()['s_' + str(j-1)][i])
                    except:
                        print("Fail d_m")
                        break

        # cons. 7~8
        for j in range(iter_):
            try:
                m.addConstr(quicksum(globals()['d_m_' + str(j)][i] for i in range(self.Total_map)) <= self.D_size)
                # Dowload size
                m.addConstr(quicksum(globals()['s_' + str(j)][i] for i in range(self.Total_map)) <= self.B_size, "c_size")
            except:
                print("Fail sum")

        # m.feasRelaxS(0, True, False, True)
        # m.setParam('TimeLimit', 60)
        # m.setParam('OutputFlag', 0)

        m.optimize()
        print(m.display())

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
        print("Gap: \n", m.Params.MIPGap)
        a = [[0 for _ in range(self.Total_map)] for _ in range(iter_)]
        it = 0
        pos = 0
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))
            if v.varName[0] == "s":
                a[it][pos] = v.x
                pos += 1
                if pos >= self.Total_map:
                    pos = 0
                    it += 1

        print('Obj: %g' % m.objVal)

        return a# The update of t_minus_x is the same meaning of action
