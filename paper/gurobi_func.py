from gurobipy import *


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
        self.D_size = 40
        self.B_size = 50
        # self.action = [0 for _ in range(794)]
        self.M = 0.5
        self.time_round = 0

    def MIP_formulation(self):
        # Create a new model
        m = Model("mip1")

        # Create variables
        s_ = m.addVars(range(794), ub=1, lb=0, vtype=GRB.BINARY, name='s_')
        x_ = m.addVars(range(794), ub=1, lb=0, vtype=GRB.BINARY, name='re_s_')
        hit = m.addVars(range(625), ub=1, lb=0, vtype=GRB.BINARY, name='hit_')
        fetch = m.addVars(range(625), ub=1, lb=0, vtype=GRB.BINARY, name='fetch_')
        d_m_ = m.addVars(range(794), ub=1, lb=0, vtype=GRB.BINARY, name='d_m_')
        y_h = m.addVars(range(16), name="y_h")
        z_h = m.addVars(range(16), vtype=GRB.BINARY, ub=1, lb=0, name="z_h")

        # Integrate new variables
        m.update()

        # Set objective
        m.setObjective(quicksum(self.vec_num[i] * hit[i] * 1 for i in range(625)) +
                       quicksum(self.vec_num[i] * fetch[i] * 10 for i in range(625)) +
                       quicksum(y_h[i] * self.f2_w[i] for i in range(16)) + self.f2_b, GRB.MINIMIZE)

        # Set constraints
        # cons. 1~4

        for i in range(16):  # 16:H
            try:
                # 求出y_h max， f1_w 正負，x_設 {0, 1}
                m.addConstr(y_h[i] >= quicksum(self.f1_w[i][j] * x_[j] for j in range(794)) + self.f1_b[i], "c1_")
                # self.M_minus =

                m.addConstr(
                    y_h[i] <= quicksum(self.f1_w[i][j] * x_[j] for j in range(794)) + self.f1_b[i] - self.M_minus * (
                            1 - z_h[i]), "c2_")
                m.addConstr(y_h[i] <= self.M_plus * z_h[i], "c3_")
                # m.addConstr(y_h[i] >= 0)

            except:
                print("Error")

        # cons. x_

        for i in range(794):
            try:
                # m.addConstr((s_[i] == 0) >> (x_[i] == 1))
                # m.addConstr((s_[i] == 1) >> (x_[i] == 0))
                m.addConstr(x_[i] == 1 - s_[i], "cx_")
            except:
                print("Fail")


        # cons. Miss,hit  Miss,fetch
        for i in range(625):
            try:
                m.addConstr(hit[i] == and_(x_[i], s_[625 + getBigmap(i)]), "hit_")  # (m is not) and (M is) <hit>
                m.addConstr(fetch[i] == and_(x_[i], x_[625 + getBigmap(i)]), "fetch_")
            except:
                print("Fail H F.")
                sys.exit()

        # Download parameters

        # cons. 5~7
        for i in range(794):
            try:
                m.addConstr(d_m_[i] >= s_[i] - self.t_minus_s_[i])

            except:
                print("Fail d_m")
                break


        # cons. 7~8
        try:
            m.addConstr(quicksum(d_m_[i] for i in range(794)) <= self.D_size)  # Dowload size
            m.addConstr(quicksum(s_[i] for i in range(794)) <= self.B_size, "c_size")
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

        len = 0
        for v in m.getVars():
            if len < 794:
                self.t_minus_s_[len] = int(v.x)
                len += 1
                print('%s %g' % (v.varName, v.x))
            else:
                break
        print('Obj: %g' % m.objVal)

        return self.t_minus_s_  # The update of t_minus_x is the same meaning of action
