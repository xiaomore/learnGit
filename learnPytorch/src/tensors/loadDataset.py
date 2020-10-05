# -*- coding: utf-8 -*-
# @Time : 2020/10/4 17:51
# @Author : Heng LI
# @FileName: loadDataset.py
# @Software: PyCharm

import torch
import numpy as np
from torch.utils.data import Dataset  # Dataset是个抽象类
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):  # 继承自Dataset
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  # 可以根据索引获取每行的数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据集条数
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')

train_loader = DataLoader(dataset=dataset,  # 数据集
                          batch_size=32,    # 每一次处理的数据集大小
                          shuffle=True,     # 是否把数据集打乱
                          num_workers=2)    # 是否使用多个线程，Windows下设置为0， Linux可以>0


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入8维，输出6维
        self.linear2 = torch.nn.Linear(6, 4)  # 上一步的输出当作这一步的输入，输出4维
        self.linear3 = torch.nn.Linear(4, 1)  # 上一步的输出当作这一步的输入，输出1维
        self.sigmoid = torch.nn.Sigmoid()     # 调用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. prepare dataset
        inputs, labels = data

        # 2. design model
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # 3. backward
        optimizer.zero_grad()
        loss.backward()

        # 4. update
        optimizer.step()

