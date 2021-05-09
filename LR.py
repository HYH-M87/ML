import numpy as np

class my_LR():
    def __init__(self):
        self.a = None 
        self.b = None
        self.coef_ = None              #特征权重向量
        self.interception_ = None  #截距
        self.theta = None
    
    #简单线性回归
    def fit(self,X_train,Y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(Y_train)
        self.a = (X_train- x_mean).dot(Y_train-y_mean)/ (X_train- x_mean).dot (X_train- x_mean)
        self.b = y_mean - self.a * x_mean

    def predict(self,X):
        return self.a * X + self.b
    
    #多元线性回归
    def fit_normal(self,X_train,Y_train):
        Xb = np.hstack([np.ones((len(X_train),1)),X_train])
        self.theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(Y_train)
        self.interception_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
            
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self
        
    

    
    
    def predict_normal(self,X_test):
        X_test = np.hstack([np.ones((len(X_test),1)),X_test])
        return X_test.dot(self.theta)
    
    def MSE(self,y_ture,y_predict):
        return np.sum((y_ture - y_predict)**2) / len(y_ture)
    
    def RMSE(self,y_ture,y_predict):
        return np.sqrt(np.sum((y_ture - y_predict)**2) / len(y_ture))
    
    def MAE(self,y_ture,y_predict):
        return np.sum(np.abs(y_ture - y_predict)) / len(y_ture)
    
    def R_square(self,y_ture,y_predict):
        return 1 - self.MSE(y_ture,y_predict) / np.var(y_ture)
    
    def score(self,x_test,y_test):
        y_predict = self.predict_normal(x_test)
        return self.R_square(y_test , y_predict)