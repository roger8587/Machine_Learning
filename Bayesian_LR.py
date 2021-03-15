import numpy as np

class Bayesian_Linear_Reg(object):
    def __init__(self, Data, Y):
        ones = np.ones(Data.shape[0], dtype=int).reshape(-1, 1)
        self.X = np.concatenate([ones, Data], axis=1)
        self.Y = Y
        
    def posterior(self, alpha, beta):
        self.beta = beta
        self.S_N_inv = alpha * np.eye(self.X.shape[1]) + beta * self.X.T.dot(self.X)
        self.S_N = np.linalg.inv(self.S_N_inv)
        self.m_N = beta * self.S_N.dot(self.X.T).dot(self.Y)

    def posterior_predictive(self, Phi_test):
        y = Phi_test.dot(self.m_N)
        y_var = 1 / self.beta + np.sum(Phi_test.dot(self.S_N) * Phi_test, axis = 1)
        return y, y_var
