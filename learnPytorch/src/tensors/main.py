# -*- coding: utf-8 -*-
# @Time : 2020/10/5 17:21
# @Author : Heng LI
# @FileName: main.py
# @Software: PyCharm
from tensors.miltiClassification import train, test

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()