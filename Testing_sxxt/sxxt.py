import numpy as np
import random

formal_list = [[0 for _ in range(2)] for _ in range(10)]
for i in range(10):
    formal_list[i][1] = random.randint(0,10)
print(formal_list)

a_ = sorted(formal_list, key=lambda s: s[1])

print(a_)