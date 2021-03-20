import numpy as np 

class LassoRegression() : 
    def __init__( self, learning_rate, iterations, l1_penality ) : 
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.l1_penality = l1_penality 
    def fit( self, X, Y ) : 
        self.m, self.n = X.shape 
        self.W = np.zeros( self.n ) 
        self.b = 0
        self.X = X 
        self.Y = Y 
        # gradient descent learning 
        for i in range( self.iterations ) : 
            self.update_weights() 
        return self
    def update_weights( self ) : 
        Y_pred = self.predict( self.X ) 
        # calculate gradients   
        dW = np.zeros( self.n ) 
        for j in range( self.n ) : 
            if self.W[j] > 0 : 
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )  
                         + self.l1_penality ) / self.m 
            else : 
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )  
                         - self.l1_penality ) / self.m 
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
        # update weights 
        self.W = self.W - self.learning_rate * dW 
        self.b = self.b - self.learning_rate * db 
        return self
      
    def predict( self, X ) : 
        return X.dot( self.W ) + self.b 
