import sys
import random
import numpy as np
import torch as t
import torch.nn as nn
import operator as op

loss_func = nn.MSELoss()

input = [[1.0], [2.0]]
input = t.FloatTensor(input)
input2 = [[1.0], [2.1]]
input2 = t.FloatTensor(input2)

loss = loss_func(input, input2)
print(loss)

input3 = [1.0, 2.0]
input3 = t.FloatTensor(input3)
input4 = [1.0, 2.1]
input4 = t.FloatTensor(input4)

loss = loss_func(input3, input4)
print(loss)




