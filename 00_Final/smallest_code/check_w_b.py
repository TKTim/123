import random
import sys
import torch
import torch.nn as nn
import numpy as np
import gurobi_INT as gb
import logging
import time
import os
import signal
import time
from torch.autograd import Variable

Total_map_number = 97

input_dim = Total_map_number  # Q() = S x A
hidden_dim = 16
output_dim = 1


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim_in):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim_in
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_dim)

        # self.sigmoid = torch.nn.ReLU()
        # self.to(device=cuda0)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output


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

    # print("fc2_weight_temp:", fc2_weight_temp)
    # print("fc2_bias_temp:", fc2_bias_temp)

    # 0: fc1.weight: torch.Size([16, 794])
    # 1: fc1.bias: torch.Size([16])
    # 2: fc2.weight: torch.Size([1, 16])
    # 3: fc2.bias: torch.Size([1])

    fc2_weight = [0.0 for _ in range(hidden_dim)]
    for i in fc2_weight_temp:
        for j in range(hidden_dim):
            fc2_weight[j] = i[j]
    fc2_bias = fc2_bias_temp[0]

    print("fc1_weight:", fc1_weight)
    print("fc1_bias:", fc1_bias)

    print("fc2_weight:", fc2_weight)
    print("fc2_bias:", fc2_bias)


dict_path = sys.argv[1]

testing_net = Net(input_dim, hidden_dim, output_dim)
try:
    testing_net.load_state_dict(torch.load(dict_path))
    print("Loaded file success.")
except FileNotFoundError:
    print("No saved files.")

get_Weight_and_Bias(testing_net)
