import pandas as pd
import random 
import matplotlib.pyplot as plt
random.seed(42)
from logru import logging

import numpy as np

X = np.linspace(0,10,100)
y = X * 2.14 + 6 + np.random.randn(100) * 2

plt.scatter(x=X,y=y,label='data')

def fit(X,y,num_iter = 100,lr = 1e-4):
    w,b = 0.01,0.01
    n = len(X)
    for i in range(num_iter):
        loss = sum((y - (w * X + b))^2) / float(n)
        tmp = sum((y - (w * X + b))*X )
        tmp2 = sum(y - (w * X + b) )

        w = w - lr * (tmp * (-2/float(n)) )
        b = b - lr * (tmp2 * (-2/float(n)))
        
        
    return w,b
for x,c in zip([100,500,1000,2000],['red','blue','green','black']):
    w,b = fit(X,y,x,1e-4)
    pred_y = w * X + b
    plt.plot(X,pred_y,color=c,label=f'fit_{x}')

plt.legend()
plt.savefig('./ML/线性回归/最小二乘法_梯度下降.png')
plt.show()