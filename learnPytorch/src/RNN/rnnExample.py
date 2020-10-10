# -*- coding: utf-8 -*-
# @Time : 2020/10/10 17:14
# @Author : Heng LI
# @FileName: rnnExample.py
# @Software: PyCharm

import torch

input_size = 4
batch_size = 1
hidden_size = 4
num_layers = 1

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]
# print(x_one_hot)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# print(inputs.size())
# print(inputs)

# labels = torch.Tensor(y_data).view(-1, 1)
labels = torch.LongTensor(y_data)
# print(labels.size())
# print(labels)

class RnnModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RnnModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.rnn = torch.nn.RNN(input_size=self.input_size,  # 与RNNCell不同的是，需要传num_layers参数，表示RNN的层数
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


model = RnnModel(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    # print('idx:', type(idx))  # idx类型为Tensor
    idx = idx.data.numpy()  # 转成numpy.ndarray
    # print('idx type:', type(idx))  # idx type: <class 'numpy.ndarray'>
    print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch[%d/15] loss=%.4f' % (epoch+1, loss))