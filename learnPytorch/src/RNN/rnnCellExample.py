# -*- coding: utf-8 -*-
# @Time : 2020/10/10 16:21
# @Author : Heng LI
# @FileName: rnnCellExample.py
# @Software: PyCharm

# 参考课程：B站 刘老师《PyTorch深度学习实践》完结合集 https://www.bilibili.com/video/BV1Y7411d7Ys?p=12

import torch

"""
假如输入x = [[1, 2, 3],
           [3, 4, 5],
           [6, 7, 8],
           [9, 10, 11]]
           
令 x = [x1, x2, x3, x4], 其中 x1 = [1, 2, 3]， 
                         x2 = [3, 4, 5], 
                         x3 = [6, 7, 8]
                         x4 = [9, 10, 11]
那么：
    input_size=3， 指x1, x2, x3, x4分别有三个元素
    seq_size=4, 指x1, x2, x3, x4共四个
"""
input_size = 4  # 输入大小
batch_size = 1  # 批处理大小
hidden_size = 4  # 隐藏大小：这里其实指的是，每一个输出的y的大小

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]
print(x_one_hot)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
print(inputs.size())
print(inputs)

labels = torch.Tensor(y_data).view(-1, 1)
print(labels.size())
print(labels)

class RnnCellModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(RnnCellModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnnCell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnnCell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


model = RnnCellModel(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label.long())
        _, idx = hidden.max(dim=1)
        print(idx2char[idx], end='')

    loss.backward()
    optimizer.step()
    print(', Epoch[%d/15] loss=%.4f' % (epoch+1, loss))