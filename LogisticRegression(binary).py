def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class LogisticRegression():    
    np.random.seed(0)
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate 
        self.iterations = iterations
    
    def fit(self, X, t):
        self.X = X
        self.t = t
        self.m, self.n = X.shape
        self.w = np.random.randn(self.n)
        self.b = np.random.randn(1)
        for i in range(self.iterations):
            error = self.cross_entropy(self.X, self.t)
            if (i+1) % 10 == 0:
                print('Iteration{}/{}, Cross Entropy Error: {:.3f}'.format(i+1,self.iterations,error))
            self.update_weights()
    def update_weights( self ) : #gradient decent
        scale = sigmoid(self.y(self.X))-self.t
        dw = scale.T.dot(self.X)/self.m
        db = np.mean(scale,0)
        self.w -= self.learning_rate * dw 
        self.b -= self.learning_rate * db
    
    def y(self, X):
        return self.w.dot(X.T) + self.b
    
    def cross_entropy(self, X, t):
        s = sigmoid(self.y(X))
        error = -np.sum(t * np.log(s) + (1 - t) * np.log(1-s))
        return error/self.m
    
    def predict(self, X):
        Y = sigmoid(self.y(X))
        pre = Y>=1/2
        return pre*1
