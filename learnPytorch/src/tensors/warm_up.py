# -*- coding: utf-8 -*-
# @Time : 2020/10/2 17:12
# @Author : LI Heng
# @FileName: warm_up.py
# @Software: PyCharm

import numpy as np

# N是批处理大小，D_in是输入维度，H是隐藏层维度，D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机生成输入数据，为64*1000的矩阵（randn()）
x = np.random.randn(N, D_in)
# 随机生成输出数据，为64*10的矩阵（randn()）
y = np.random.randn(N, D_out)

# 随机生成权重矩阵（w不是固定不变的，下面for循环里会进行训练，不断的修正w的值，以使预测值和真实值不断接近）
# w1为1000*100的矩阵，w2为100*10的矩阵
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# 学习率，也称之为步长，为了防止权重值一次性改变太多或太少
learning_rate = 1e-6

# 循环训练500次
for t in range(500):
    # 正向传播的过程：计算预测值y_pred
    # h表示隐藏层，这里dot是矩阵的点积(第一个矩阵的第一行与第二个矩阵的第一列对应元素分别相乘再相加，为新矩阵的第一个位置的值，其他的行和列同理可得)；
    # 其中x:64*1000，w1:1000*100，点积后得到的h:64*100
    h = x.dot(w1)
    # 这里的激活函数仅仅简单的使用了与0相比的最大值，maximum(h, 0)表示h中的元素与0相比，取大--其实就是h中的负值改为0
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    print(t, loss)

    #后向传播过程：主要简化求导
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
