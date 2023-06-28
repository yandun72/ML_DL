import numpy as np
'''
梯度推导
https://blog.csdn.net/u012494321/article/details/104446559
'''
class LogisticRegression:
    def __init__(self,X,y,lr,num_iter,logging_step):
        self.X = np.array(X) #n,m
        self.y = np.array(y) #n,1
        self.dim = X.shape[1] 
        self.lr = lr
        self.num_iter = num_iter
        self.weight = np.array([0.01 for i in range(self.dim)]) #m,1
        self.bias = -0.01
        self.n_sample = len(X)
        self.logging_step = logging_step
        
    def fit(self):

        for i in range(self.num_iter):
            pred_y =  self.sigmoid(np.dot(self.X,self.weight) + self.bias)

            loss = -(1 / float(self.n_sample)) * np.sum( self.y * np.log(pred_y) + (1-self.y) * np.log(1 - pred_y)  )

            if i % self.logging_step == 0:
                print(f'step:{i},loss:{round(loss,3)}')
            dw =   (1 / float(self.n_sample)) * np.dot(self.X.T,(pred_y - self.y)) 
            db =   (1 / float(self.n_sample)) * np.sum( pred_y - self.y )

            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,data):
        return self.sigmoid(  np.dot(data,self.weight) + self.bias )

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

