from gurobipy import *
import random
m = Model("mip1")


# Create variables
a = m.addVar(ub=1, lb=0, vtype=GRB.INTEGER, name="a")
s_ = m.addVars(range(10), ub=1, lb=0, vtype=GRB.BINARY, name='s_')
m.update()

# Set objective
m.setObjective(a + quicksum(s_[i] for i in range(10)), GRB.MAXIMIZE)

# Set constraints
m.addConstr(a >= 3, "a")
m.addConstr(a <= 2, "a2")

m.optimize()
m.write('mip1.lp')

print(m.display())

try:
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
except:
    m.optimize()
    m.write('mip1.lp')
    print(m.display())
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
