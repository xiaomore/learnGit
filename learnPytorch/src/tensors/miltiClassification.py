# -*- coding: utf-8 -*-
# @Time : 2020/10/5 16:46
# @Author : Heng LI
# @FileName: miltiClassification.py
# @Software: PyCharm

# import package
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


# Step1: prepare dataset
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.1307, ), (0.3081, ))  # Mnist数据集的均值和方差
])

train_dataset = datasets.MNIST(root='../../../dataset',  # 数据集存放的目录
                               train=True,               # 是否为训练集
                               download=True,            # 是否下载
                               transform=transform)      # 标准化

train_loader = DataLoader(dataset=train_dataset,         # 数据加载器；指明加载的数据集
                          shuffle=True,                  # 是否打乱数据集
                          batch_size=batch_size)         # 每一次处理的数据集大小

test_dataset = datasets.MNIST(root='../../../dataset',
                              train=False,
                              transform=transform,
                              download=True)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# Step2: design model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)  # 784 = 1 * 28 * 28，
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)    # 10表示输出10维，代表最终分类的数目

    def forward(self, x):
        x = x.view(-1, 784)      # 相当于reshape，
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)        # 损失函数使用CrossEntropyLoss()，内部已经进行log和softmax运算，所以最后一层不用激活函数
                                 # 若损失函数使用NLLLoss()，随后一层也要使用激活函数


model = Net()

# Step3: construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()  # 损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 优化器

# Step4: train and test
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # 1. 预测
        outputs = model(inputs)
        # 2. 计算损失
        loss = criterion(outputs, target)

        # 3. 反向
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        running_loss += loss
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss : %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))
