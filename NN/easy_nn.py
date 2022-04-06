from sklearn.datasets import load_boston
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X, y = data, target

#  parameters
input_dim = 13
output_dim = 1
hidden_dim = 13
epoch = 50000


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, num_classes)
        # self.sigmoid = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        return output



#  cuda
if torch.cuda.is_available():
    cuda0 = torch.device('cuda:0')
    print(cuda0)

X_t = torch.tensor(X, dtype=torch.float32, device=cuda0)
Y_t = torch.tensor(y, dtype=torch.float32, device=cuda0)

net = Feedforward(input_dim, hidden_dim, output_dim)

loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.to(cuda0)

loss_all = []

for i in range(epoch):
    output = net(X_t)
    loss = loss_func(output.flatten(), Y_t.flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_all.append(loss.item())

    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}/{epoch}, loss = {loss.item():.4f}')

torch.save(net.state_dict(), './saved/net_params_v2.pkl')

net.eval()
running_accuracy = 0
total = len(Y_t)
with torch.no_grad():

    for i in range(total):
        predicted_value = net(X_t[i])
        true_value = Y_t[i]
        if abs(true_value - predicted_value) <= 5:
            running_accuracy += 1

    acc = running_accuracy / total
    print('acc: ', acc)
