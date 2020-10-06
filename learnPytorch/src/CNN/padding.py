# -*- coding: utf-8 -*-
# @Time : 2020/10/6 15:22
# @Author : Heng LI
# @FileName: padding.py
# @Software: PyCharm

# 参考课程：B站 刘老师《PyTorch深度学习实践》完结合集 https://www.bilibili.com/video/BV1Y7411d7Ys?p=10

import torch

input = [3, 4, 5,
         2, 4, 6,
         1, 6, 7,
         9, 7, 4]   # 必须要是个方阵吗？--不是必须方阵，现实数据集中，也无法保证都是方阵

# 输入转为张量，1, 1, 4, 3表示：batch(批处理的大小), channel(通道数，其实就是输入的矩阵数量), wight(输入的矩阵的行数), high(输入的矩阵的列数)
input = torch.Tensor(input).view(1, 1, 4, 3)

# 卷积层：1, 1表示：输入时1个通道，输出是1个通道，kernel_size是指filter矩阵的维度，padding=1是输入的填充（就是在输入矩阵最外层增加一圈，并填充为0）；bias是偏差
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

# kernel：这里的kernel指的就是与输入矩阵做点乘的那个filter矩阵，先转为张量，在塑形为3 * 3的矩阵，1,1分别表示kernel的输入和输出的channel，3,3表示kernel为3*3
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)

# 把kernel的数据赋给卷积层的权重
# 这里的意思就是：Filter矩阵其实就是权重矩阵，输入矩阵中每一个块与Filter矩阵点乘，得到输出矩阵中的一个元素的值。
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)