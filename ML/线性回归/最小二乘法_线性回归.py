import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  random import random

# 设置随机种子，以确保结果可重复
np.random.seed(42)

# 生成输入特征 X，这里以一个简单的一维特征为例
X = np.linspace(0, 10, 100)  # 生成 0 到 10 之间的 100 个数据点

# 生成目标变量 y，假设存在线性关系 y = 2*X + 3，并添加一些噪声
y = 2 * X + 3 + np.random.randn(100) * 2 #np.random.randn 生成正态分布的数据

# 可视化生成的数据
plt.scatter(X,y,label='data')

k = ( sum(X*y) - y.mean()*X.sum() ) / (sum(X*X) - X.mean()*X.sum())
b = y.mean() - k * X.mean()
pred_y = X*k + b
plt.plot(X,pred_y,label='fit',color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./ML/线性回归/最小二乘法.png')
plt.show()
