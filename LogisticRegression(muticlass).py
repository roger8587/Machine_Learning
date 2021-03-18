import numpy as np

def get_onehot(y):
    classes = np.unique(y)
    one_hot = np.zeros((len(y),len(classes)))
    for i in range(len(y)):
        one_hot[i][y[i]] = 1
    return one_hot

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X),1).reshape(-1,1)

class LogisticRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate 
        self.iterations = iterations
    
    def fit(self, X, t):
        self.X = X
        self.t = t
        self.oh_t = get_onehot(t)
        self.classes = len(np.unique(t))
        self.m, self.n = X.shape
        self.w = np.random.randn(self.classes,self.n)
        self.b = np.random.randn(self.classes)
        for i in range(self.iterations):
            error = self.cross_entropy(self.X, self.t)
            if (i+1) % 10 == 0:
                print('Iteration{}/{}, Cross Entropy Error: {:.3f}'.format(i+1,self.iterations,error))
            self.update_weights()
    def update_weights( self ) : #gradient decent
        scale = softmax(self.y(self.X))-self.oh_t
        dw = scale.T.dot(self.X)/self.m
        db = np.mean(scale,0)
        self.w = self.w - self.learning_rate * dw 
        self.b = self.b - self.learning_rate * db
    
    def y(self, X):
        return self.w.dot(X.T).T + self.b
    
    def cross_entropy(self, X, t):
        s = softmax(self.y(X))
        oh_t = get_onehot(t)
        error = -np.sum(np.log(s)*oh_t)
        return error/self.m
    
    def predict(self, X):
        Y = softmax(self.y(X))
        pre = np.argmax(Y,1)
        return pre
