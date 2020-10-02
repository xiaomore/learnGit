# -*- coding: utf-8 -*-
# @Time : 2020/10/2 17:12
# @Author : LI Heng
# @FileName: warm_up.py
# @Email : liheng00666@163.com
# @Software: PyCharm

import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
