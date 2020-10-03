# -*- coding: utf-8 -*-
# @Time : 2020/10/3 15:20
# @Author : Heng LI
# @FileName: backPro.py
# @Software: PyCharm

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0])
w1.requires_grad = True

w2 = torch.tensor([1.0])
w2.requires_grad = True

b = torch.tensor([1.0])
b.requires_grad = True


def forward(x):
    return x ** 2 * w1 + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y).pow(2)


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        w1.data -= 0.01 * w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.01 * b.grad.data

        print("w1:", w1.item(), "w2:", w2.item(), "b:", b.item())

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("process:", epoch, l.item())
    print("*"*20)