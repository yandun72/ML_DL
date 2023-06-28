from LogisticRegression import LogisticRegression
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

n_sample = 2000

#生成第一个正太分布的数据
mean1 = np.array([2,2])
sigma = np.array([[1,0.5],[0.5,1]])
feat1 = np.random.multivariate_normal(mean1,sigma,n_sample)


#生成第二个正太分布的数据
mean2 = np.array([-1.5,0])
sigma2 = np.array([[1,0.5],[0.5,1]])
feat2 = np.random.multivariate_normal(mean2,sigma2,n_sample)

#绘制散点图
plt.scatter(feat1[:,0],feat1[:,1],color='red',label='0')
plt.scatter(feat2[:,0],feat2[:,1],color='blue',label='1')

#合并数据
feat = np.vstack((feat1,feat2))

label = np.array([0] * len(feat1) + [1] * len(feat2))

lr = 5e-3
num_iter = 2000
logging_step = num_iter // 4
model = LogisticRegression(feat,label,lr,num_iter,logging_step)
model.fit()

# #二维数据，可以绘制决策边界直线,决策边界的方程为w1 * x1 + w2 * x2 = 0
weight,bias = model.weight,model.bias

x1 = np.linspace(-4,4,100)
x2 = weight[0] * x1 / weight[1]
print(weight,bias)

plt.plot(x1,x2,color = 'black',label='descision bound')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('./ML/逻辑回归/逻辑回归.png')
plt.show()
'''
step:0,loss:0.707
step:500,loss:0.224
step:1000,loss:0.174
step:1500,loss:0.153
[-1.56934786 -0.41634423] 0.39075077028333566
'''