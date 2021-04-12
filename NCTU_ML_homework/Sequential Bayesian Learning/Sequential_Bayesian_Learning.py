import numpy as np 

class Bayesian_Learning(object):
    def __init__(self, Data, Y, alpha, beta):
        self.X = Data
        self.Y = Y
        self.beta = beta #給定
        self.alpha = alpha #給定10**-6
        self.prior_mean = np.zeros(self.X.shape[1])
        self.prior_cov = self.alpha * np.eye(self.X.shape[1])
        
    def posterior(self):
        self.S_N_inv = self.prior_cov + self.beta * self.X.T.dot(self.X)
        self.S_N = np.linalg.inv(self.S_N_inv)
        self.m_N = self.S_N.dot(self.prior_cov.dot(self.prior_mean).reshape(-1,1) + self.beta * self.X.T.dot(self.Y))
        self.prior_mean = self.m_N
        self.prior_cov = self.S_N
        return self.m_N, self.S_N

    def posterior_predictive(self, X_test):
        y = X_test.dot(self.m_N)
        y_var = 1 / self.beta + np.sum(X_test.dot(self.S_N) * X_test)
        return y, y_var