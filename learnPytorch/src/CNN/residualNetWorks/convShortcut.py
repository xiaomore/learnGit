# -*- coding: utf-8 -*-
# @Time : 2020/10/10 10:51
# @Author : Heng LI
# @FileName: convShortcut.py
# @Software: PyCharm

# 参考课程：B站 刘老师《PyTorch深度学习实践》完结合集 https://www.bilibili.com/video/BV1Y7411d7Ys?p=11

# Step0: import package
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

# Step1: prepare dataset
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../../../../dataset',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_dataset = datasets.MNIST(root='../../../../dataset',
                              train=False,
                              transform=transform,
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

# Step2: design model
class ConvShortcut(nn.Module):
    def __init__(self, channels):
        super(ConvShortcut, self).__init__()
        self.channels = channels

        self.conv1x1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        # print(x.size())
        x1 = self.conv1x1(x)
        # print(x1.size())
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        # print(y.size())
        x = F.relu((x1 + y) * 0.5)
        return x


class ConvShortcutNet(nn.Module):
    def __init__(self):
        super(ConvShortcutNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.mp = nn.MaxPool2d(2)

        self.cBlock1 = ConvShortcut(16)
        self.cBlock2 = ConvShortcut(32)

        self.fc = nn.Linear(512,  10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.cBlock1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.cBlock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = ConvShortcutNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step3: construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step4: train and test
def train_resNet(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # 1. 预测
        output = model(inputs)
        # 2. 计算损失
        loss = criterion(output, target)
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d %5d] loss: %.3f" % (epoch, batch_idx, running_loss / 2000))
            running_loss = 0.0

def tst_resNet():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            input, target = data
            input, target = input.to(device), target.to(device)
            output = model(input)
            _, predicted = torch.max(output, dim=1)

            correct += (predicted == target).sum().item()
            total += target.size(0)
    print("Accuracy is %d%% [%d/%d]: " % (100 * correct / total, correct, total))

if __name__ == "__main__":
    for epoch in tqdm(range(10)):
        train_resNet(epoch)
        tst_resNet()


