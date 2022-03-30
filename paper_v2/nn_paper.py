import torch.nn as nn
import torch
import torch.utils.data as Data
import sys
import numpy

'''

class DataLoader:
    def __init__(self):
        pass

    def get_task_batch(self):
        pass

    def get_iterator(self):
        tnt_dataset = torchnet.dataset.ListDataset(
            elem_list=range(self.task_num), load=self.get_task_batch)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            # 此函数可以使每个worker使用不同的随机种子
            worker_init_fn=self.worker_init_fn_seed,
            shuffle=(False if self.test else True))
        return data_loader

    def worker_init_fn_seed(self, worker_id):
        seed = 10 + 5 * worker_id
        np.random.seed(seed)

    def __call__(self):
        return self.get_iterator()
'''


def train(epochs, loader, model, optimizer):
    loss_all = []
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # 梯度清零
            model.zero_grad()
            tag_scores = model(batch_x)
            loss = loss_func(tag_scores.flatten(), batch_y.flatten())

            # 后向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # record loss
            # loss_all.append(loss.item())

            if epoch % 5 == 0 and step % 3 == 0:
                print(f'epoch {epoch}, loss = {loss:.4f}')
            '''
            if  step % 3 == 0:
                print(f'loss = {loss:.4f}')'''


def getWeightandBias(model):
    num = 0
    for par in model.parameters():
        if num == 0:
            fc1_weight = par.cpu().detach().numpy()
        elif num == 1:
            fc1_bias = par.cpu().detach().numpy()
        elif num == 2:
            fc2_weight = par.cpu().detach().numpy()
        else:
            fc2_bias = par.cpu().detach().numpy()
        num += 1
        # 0: fc1.weight: torch.Size([16, 794])
        # 1: fc1.bias: torch.Size([16])
        # 2: fc2.weight: torch.Size([1, 16])
        # 3: fc2.bias: torch.Size([1])
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    py_text = sys.argv[1]
    print(py_text)

    f = open("Reward.txt", 'r')

    lines = (line.rstrip() for line in f.readlines())  # All lines including the blank ones. Skip first line.
    lines = (line for line in lines if line)  # Non-blank lines

    # The file is 794 line, but we need to separate them.
    data = [[0 for _ in range(794)] for _ in range(1000)]  # ignore the last target_value
    target = [0.0 for _ in range(1000)]

    # Read file
    iter_time = 0
    for line in lines:
        search_ = False
        search_reward = False
        map_pos = 0
        temp_reward = ""
        for i in line:
            if i == "i":  # ignore iter line
                break
            if search_:  # Starting to search everything
                if search_reward:  # Search reward on the last item
                    temp_reward += i
                    if i == " ":  # The end of the line
                        target[iter_time] = float(temp_reward)
                        iter_time += 1
                        search_ = False
                        search_reward = False
                        break
                elif i == "-":
                    search_reward = True
                    temp_reward += i
                elif i == " ":
                    continue
                else:
                    if i == "1":
                        data[iter_time][map_pos] = 1
                    else:
                        data[iter_time][map_pos] = -1
                    map_pos += 1
            if i == "[":
                search_ = True

    X, y = data, target

    #  Test to see if the data is all right
    for i in range(5):
        print(X[i])
    for i in range(5):
        print(y[i])

    #  parameters
    input_dim = 794
    output_dim = 1
    hidden_dim = 16
    batch_size = 500
    epochs = 20


    class Feedforward(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, batch_size):
            super(Feedforward, self).__init__()
            self.batch_size = batch_size
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
    else:
        cuda0 = torch.device('CPU')

    X_t = torch.tensor(X, dtype=torch.float32, device=cuda0)
    Y_t = torch.tensor(y, dtype=torch.float32, device=cuda0)
    torch_dataset = Data.TensorDataset(X_t, Y_t)

    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  #
        num_workers=0,  # Multi thread for loading data
    )

    model = Feedforward(input_dim, hidden_dim, output_dim, batch_size)

    loss_func = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(cuda0)

    # Training process
    train(epochs, loader, model, optimizer)

    # Saved
    torch.save(model.state_dict(), './saved/net_params_v2.pkl')

    model.eval()
    running_accuracy = 0
    total = len(Y_t)
    with torch.no_grad():
        for i in range(total):
            predicted_value = model(X_t[i])
            true_value = Y_t[i]
            if abs(true_value - predicted_value) <= 5:
                running_accuracy += 1

        acc = running_accuracy / total
        print('acc: ', acc)

    testing = X_t[10]
    predicted_testing = model(testing)
    na = predicted_testing.cpu().detach().numpy()
    print("predicted: ", na)

    for name, para in model.named_parameters():
        print('{}: {}'.format(name, para.shape))

    # Get Weight and Bias
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = getWeightandBias(model)

    print(fc1_weight)
    print(fc1_bias)
    print(fc2_weight)
    print(fc2_bias)
'''
    from torchsummaryX import summary

    # summary(model, torch.zeros(5, 3, 512, 512))
    for name in model.state_dict():
        print(" %s  :  %s" % (name, model.state_dict()[name].shape))
'''
