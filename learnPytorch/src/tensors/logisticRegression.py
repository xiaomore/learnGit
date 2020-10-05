# -*- coding: utf-8 -*-
# @Time : 2020/10/4 14:25
# @Author : Heng LI
# @FileName: logisticRegression.py
# @Software: PyCharm

# 本部分参考了B站刘老师《PyTorch深度学习实践》完结合集

import torch
import torch.nn.functional as F

# 整个训练过程，分为以下四步
# Step1: prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])

# Step2: design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# Step3: Construct loss and optimizer
# BCELoss():二分类交叉熵
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Step4: training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = ", model.linear.weight.data)
print("b = ", model.linear.weight.data)

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("y_test = ", y_test.item())