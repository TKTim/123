import torch
import numpy as np
import pandas as pd

from sklearn import datasets

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

# base data inform and network data
input_size = 4
hidden_size = 16
num_classes = 3
num_epochs = 1000
batch_size = 4
learning_rate = 0.001


class IrisDataset(Dataset):

    # data loading
    def __init__(self):
        iris = datasets.load_iris()
        feature = pd.DataFrame(iris.data, columns=iris.feature_names)
        target = pd.DataFrame(iris.target, columns=['target'])
        iris_data = pd.concat([target, feature], axis=1)
        # Data type change and flatten targets
        self.x = torch.from_numpy(np.array(iris_data)[:, 1:].astype(np.float32))
        self.y = torch.from_numpy(np.array(iris_data)[:, [0]].astype(np.longlong).flatten())
        self.n_samples = self.x.shape[0]

    # working for indexing
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # return the length of our dataset
    def __len__(self):
        return self.n_samples


dataset = IrisDataset()


# create data spliter
def dataSplit(dataset, val_split=0.25, shuffle=False, random_seed=0):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


# base split parameters
val_split = 0.25
shuffle_dataset = True
random_seed = 42

train_sampler, valid_sampler = \
    dataSplit(dataset=dataset, val_split=val_split, shuffle=shuffle_dataset, random_seed=random_seed)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


# Feedforward Network Module
class Feedforward(torch.nn.Module):
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


model = Feedforward(input_size, hidden_size, num_classes)

# 2) loss and optimizer
learning_rate = 0.01
# Cross Entropy
criterion = nn.CrossEntropyLoss()
# adam algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 3) Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (datas, labels) in enumerate(train_loader):

        # init optimizer
        optimizer.zero_grad()

        # forward -> backward -> update
        outputs = model(datas)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        if (i + 1) % 19 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# 4) Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for datas, labels in val_loader:
        outputs = model(datas.float())
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
