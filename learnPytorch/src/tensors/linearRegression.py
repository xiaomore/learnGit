# -*- coding: utf-8 -*-
# @Time : 2020/10/3 17:12
# @Author : Heng LI
# @FileName: linearRegression.py
# @Software: PyCharm

import torch

x_data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y_data = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # Linear()里面的两个参数，分别表示输入样本的大小，输出样本的大小
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = ", model.linear.weight.data)
print("b = ", model.linear.bias.data)

x_test = torch.tensor([[4.0, 5.0]])
y_test = model(x_test)
print("y_test = ", y_test)