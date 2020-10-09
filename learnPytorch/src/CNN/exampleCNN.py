# -*- coding: utf-8 -*-
# @Time : 2020/10/6 20:03
# @Author : Heng LI
# @FileName: exampleCNN.py
# @Software: PyCharm

# 参考课程：B站 刘老师《PyTorch深度学习实践》完结合集 https://www.bilibili.com/video/BV1Y7411d7Ys?p=10

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
                               download=True,
                               transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root='../../../dataset',
                              train=True,
                              transform=transform,
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# Step2: design model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(1, 10, kernel_size=5)：1表示输入的channel数量，10表示输出的channel数量；
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)    # Conv2d:2维卷积，主要应用于图像处理，视觉; Conv1d:1维卷积用于序列模型，NLP; Conv3d:3维卷积主要用于医疗领域及视频处理
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)   # kernel_size指的是于输入矩阵做点积的Filter矩阵的维度
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)      # pooling操作
        self.fc = torch.nn.Linear(320, 10)                    # fully connected 由320 -> 10

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))  # 卷积，池化，激活
        x = F.relu(self.pooling(self.conv2(x)))  # 卷积，池化，激活
        x = x.view(bs, -1)                       # flatten
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU计算，"cuda:0"表示使用第一块GPU，多块情况下，可以设置成"cuda:1"，"cuda:2"...
model.to(device)  # 转向使用GPU

# Step3: construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Step4: train and test
def train_cnn(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # prepare dataset
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 输入数据也迁移到GPU上计算，模型和数据要放到同一块显卡上

        optimizer.zero_grad()
        # 1. 预测
        outputs = model(inputs)
        # 2. 计算损失
        loss = criterion(outputs, target)
        # 3. 反向
        loss.backward()
        # 4. 更新参数
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d %5d] loss: %.3f" % (epoch, batch_idx+1, running_loss/2000))
            running_loss = 0.0


def test_cnn():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1)  # dim=1:表示每行的最大值，0表示每列的最大值

            total += target.size(0)
            correct += (predicted == target).sum().item()

        print("Accuracy on test set: %d %% [%d/%d]" % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train_cnn(epoch)
        test_cnn()