import sys
import random
import numpy as np
import torch as t
import torch.nn as nn
import operator as op

lr = 0.5
episode = 10
lr = lr / (episode+1) ** 0.5
print(lr)





EPSILON = 0.9
decay_rate = 0.8
for i in range(5):
    EPSILON = 1 - (1 - EPSILON)*decay_rate
print("EPSILON", EPSILON)



