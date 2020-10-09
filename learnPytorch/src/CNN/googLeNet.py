# -*- coding: utf-8 -*-
# @Time : 2020/10/7 14:00
# @Author : Heng LI
# @FileName: googLeNet.py
# @Software: PyCharm

# 参考课程：B站 刘老师《PyTorch深度学习实践》完结合集 https://www.bilibili.com/video/BV1Y7411d7Ys?p=11

# Step0: import package
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# Step1: prepare dataset
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../../../dataset',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_dataset = datasets.MNIST(root='../../../dataset',
                              train=False,
                              transform=transform,
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)


# Step2: design model.
class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        # average pool + conv1x1
        self.branch_pool = torch.nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)

        self.conv5x5_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.conv5x5_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.conv3x3_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.conv3x3_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.conv3x3_3 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

    def forward(self, x):
        # average pool + conv1x1
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.conv5x5_1(x)
        branch5x5 = self.conv5x5_2(branch5x5)

        branch3x3 = self.conv3x3_1(x)
        branch3x3 = self.conv3x3_2(branch3x3)
        branch3x3 = self.conv3x3_3(branch3x3)

        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        outputs = torch.cat(outputs, dim=1)

        return outputs


class GoogLeNet(torch.nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))  # 卷积，池化，激活
        x = self.incep1(x)                  # 做Inception
        x = F.relu(self.mp(self.conv2(x)))  # 卷积，池化，激活
        x = self.incep2(x)                  # 再做Inception
        x = x.view(in_size, -1)             # flatten

        x = self.fc(x)                      # fully connected
        return x

model = GoogLeNet()

# Step3: construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step4: train and test
def train_googLeNet(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # 1. 预测
        outputs = model(inputs)

        # 2. 计算损失
        loss = criterion(outputs, target)

        # 3. 反向传播
        loss.backward()

        # 4. 更新梯度
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d %5d] loss: %.3f" % (epoch, batch_idx+1, running_loss/2000))
            running_loss = 0.0


def te_googLeNet():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)

            target += target.size(0)
            correct += (predicted == target).sum().item()

        print("Accuracy on test set: %d %% [%d/%d]" % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(10):
        train_googLeNet(epoch)
        te_googLeNet()


