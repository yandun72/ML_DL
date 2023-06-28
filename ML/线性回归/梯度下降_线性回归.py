import pandas as pd
import random 
import matplotlib.pyplot as plt
random.seed(42)
# from logru import logging

import numpy as np

X = np.linspace(0,10,100)
y = X * 2.14 + 6 + np.random.randn(100) * 2

plt.scatter(x=X,y=y,label='data')

def fit(X,y,num_iter = 100,lr = 1e-4):
    w,b = 0.01,0.01
    n = len(X)
    for i in range(num_iter):
        loss = sum((y - (w * X + b))**2) / float(n)
        tmp = sum((y - (w * X + b))*X )
        tmp2 = sum(y - (w * X + b) )

        w = w - lr * (tmp * (-2/float(n)) )
        b = b - lr * (tmp2 * (-2/float(n)))
        if i % 50 == 0:
            print(f'num_iter:{num_iter},now_iter:{i},loss:{round(loss,3)}')
        
    return w,b
for x,c in zip([100,500,1000,2000],['red','blue','green','black']):
    w,b = fit(X,y,x,1e-4)
    pred_y = w * X + b
    plt.plot(X,pred_y,color=c,label=f'fit_{x}')

plt.legend()
plt.savefig('./ML/线性回归/最小二乘法_梯度下降.png')
plt.show()
'''
num_iter:100,now_iter:0,loss:319.637
num_iter:100,now_iter:50,loss:167.342
num_iter:500,now_iter:0,loss:319.637
num_iter:500,now_iter:50,loss:167.342
num_iter:500,now_iter:100,loss:90.735
num_iter:500,now_iter:150,loss:52.189
num_iter:500,now_iter:200,loss:32.782
num_iter:500,now_iter:250,loss:23.001
num_iter:500,now_iter:300,loss:18.06
num_iter:500,now_iter:350,loss:15.553
num_iter:500,now_iter:400,loss:14.271
num_iter:500,now_iter:450,loss:13.604
num_iter:1000,now_iter:0,loss:319.637
num_iter:1000,now_iter:50,loss:167.342
num_iter:1000,now_iter:100,loss:90.735
num_iter:1000,now_iter:150,loss:52.189
num_iter:1000,now_iter:200,loss:32.782
num_iter:1000,now_iter:250,loss:23.001
num_iter:1000,now_iter:300,loss:18.06
num_iter:1000,now_iter:350,loss:15.553
num_iter:1000,now_iter:400,loss:14.271
num_iter:1000,now_iter:450,loss:13.604
num_iter:1000,now_iter:500,loss:13.246
num_iter:1000,now_iter:550,loss:13.045
num_iter:1000,now_iter:600,loss:12.922
num_iter:1000,now_iter:650,loss:12.838
num_iter:1000,now_iter:700,loss:12.775
num_iter:1000,now_iter:750,loss:12.722
num_iter:1000,now_iter:800,loss:12.674
num_iter:1000,now_iter:850,loss:12.628
num_iter:1000,now_iter:900,loss:12.585
num_iter:1000,now_iter:950,loss:12.542
num_iter:2000,now_iter:0,loss:319.637
num_iter:2000,now_iter:50,loss:167.342
num_iter:2000,now_iter:100,loss:90.735
num_iter:2000,now_iter:150,loss:52.189
num_iter:2000,now_iter:200,loss:32.782
num_iter:2000,now_iter:250,loss:23.001
num_iter:2000,now_iter:300,loss:18.06
num_iter:2000,now_iter:350,loss:15.553
num_iter:2000,now_iter:400,loss:14.271
num_iter:2000,now_iter:450,loss:13.604
num_iter:2000,now_iter:500,loss:13.246
num_iter:2000,now_iter:550,loss:13.045
num_iter:2000,now_iter:600,loss:12.922
num_iter:2000,now_iter:650,loss:12.838
num_iter:2000,now_iter:700,loss:12.775
num_iter:2000,now_iter:750,loss:12.722
num_iter:2000,now_iter:800,loss:12.674
num_iter:2000,now_iter:850,loss:12.628
num_iter:2000,now_iter:900,loss:12.585
num_iter:2000,now_iter:950,loss:12.542
num_iter:2000,now_iter:1000,loss:12.499
num_iter:2000,now_iter:1050,loss:12.457
num_iter:2000,now_iter:1100,loss:12.415
num_iter:2000,now_iter:1150,loss:12.374
num_iter:2000,now_iter:1200,loss:12.332
num_iter:2000,now_iter:1250,loss:12.291
num_iter:2000,now_iter:1300,loss:12.251
num_iter:2000,now_iter:1350,loss:12.21
num_iter:2000,now_iter:1400,loss:12.169
num_iter:2000,now_iter:1450,loss:12.129
num_iter:2000,now_iter:1500,loss:12.089
num_iter:2000,now_iter:1550,loss:12.049
num_iter:2000,now_iter:1600,loss:12.009
num_iter:2000,now_iter:1650,loss:11.97
num_iter:2000,now_iter:1700,loss:11.931
num_iter:2000,now_iter:1750,loss:11.892
num_iter:2000,now_iter:1800,loss:11.853
num_iter:2000,now_iter:1850,loss:11.814
num_iter:2000,now_iter:1900,loss:11.775
num_iter:2000,now_iter:1950,loss:11.737
'''