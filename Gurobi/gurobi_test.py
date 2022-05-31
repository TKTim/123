from gurobipy import *
import random
m = Model("mip1")

t = [random.randint(0, 1) for _ in range(5)]

for i in range(5):
    print(t[i])


# Create variables
a = m.addVar(ub=5, lb=2, vtype=GRB.INTEGER, name="a")
s_ = m.addVars(range(10), ub=1, lb=0, vtype=GRB.BINARY, name='s_')
x_ = m.addVars(range(5), ub=1, lb=0, vtype=GRB.BINARY, name='re_x_')
hit_ = m.addVars(range(5), ub=1, lb=0, vtype=GRB.BINARY, name='hit_')
g_ = m.addVars(range(5), ub=1, lb=0, vtype=GRB.BINARY, name='g_')
# Integrate new variables
m.update()

# Set objective
m.setObjective(quicksum(s_[i] for i in range(5)), GRB.MAXIMIZE)

# Set constraints
m.addConstr(a >= 3)
for i in range(5):
    re_v1_name = "re_v1_" + str([i])
    re_v2_name = "re_v2_" + str([i])
    try:
        m.addConstr((s_[i] == 0) >> (x_[i] == 1), name=re_v1_name)
        m.addConstr((s_[i] == 1) >> (x_[i] == 0), name=re_v2_name)
    except:
        print("Fail")
    # m.addConstr(hit_[i] == and_(s_[i], t[i]))

m.addConstr(quicksum(s_[i] for i in range(5)) <= 3)


m.optimize()
m.write('mip1.lp')

print(m.display())

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % m.objVal)
