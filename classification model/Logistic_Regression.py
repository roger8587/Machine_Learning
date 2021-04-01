import numpy as np

def get_onehot(y):
    classes = np.unique(y)
    one_hot = np.zeros((len(y),len(classes)))
    for i in range(len(y)):
        one_hot[i][y[i]] = 1
    return one_hot

class activater():
    def softmax(self, X):
        return np.exp(X)/np.sum(np.exp(X),1).reshape(-1,1)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def is_multi_class(self, multi_class = True):
        if multi_class:
            return self.softmax
        else:
            return self.sigmoid

class LogisticRegression():    
    np.random.seed(0)
    def __init__(self, learning_rate, iterations, multi_class = True, fit_intercept=True):
        self.learning_rate = learning_rate 
        self.iterations = iterations
        self.activater_fun = activater().is_multi_class(multi_class)
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
    
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def fit(self, X, t):
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.X = X
        self.t = t
        self.classes = t.shape[1] if self.multi_class else 1
        self.m, self.n = X.shape
        if self.multi_class:
            self.w = np.random.randn(self.classes,self.n)
        else:
            self.w = np.random.randn(self.n)
        for i in range(self.iterations):
            error = self.cross_entropy(self.X, self.t)
            if (i+1) % 10 == 0:
                print('Iteration{}/{}, Cross Entropy Error: {:.3f}'.format(i+1,self.iterations,error))
            self.update_weights()
        
    def update_weights( self ) : #gradient decent
        scale = self.activater_fun(self.y(self.X))-self.t
        dw = scale.T.dot(self.X)/self.m
        self.w -= self.learning_rate * dw 
    
    def y(self, X):
        if self.multi_class:
            return self.w.dot(X.T).T
        else:
            return self.w.dot(X.T)
        
    def cross_entropy(self, X, t):
        s = self.activater_fun(self.y(X))
        if self.multi_class:
            error = -np.sum(np.log(s)*t)
        else:
            error = -np.sum(t * np.log(s) + (1 - t) * np.log(1-s))
        return error/self.m
    
    def predict(self, X):
        Y = self.activater_fun(self.y(X))
        if self.multi_class:
            pre = np.argmax(Y,1)
        else:
            pre = Y>=1/2
            pre = pre*1
        return pre
