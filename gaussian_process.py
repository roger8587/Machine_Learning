import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.io as io
import os

data = io.loadmat('gp.mat')['x']
label = io.loadmat('gp.mat')['t']
x_train = data[0:60]
x_test = data[60:100]
t_train = label[0:60]
t_test = label[60:100]

class gaussian_process():
    def __init__(self,thetas):
        self.thetas = np.asarray(thetas).astype(float)
        
    def kernel(self, xn, xm):
        sqdist = np.sum(xn**2, 1).reshape(-1, 1) + np.sum(xm**2, 1) - 2 * np.dot(xn, xm.T)
        return  self.thetas[0] * np.exp(-0.5*self.thetas[1]*sqdist) + self.thetas[2] + self.thetas[3]*xn.dot(xm.T)
    
    def fit(self, X_train, Y_train, sigma_y = 1):
        self.X_train = X_train
        self.Y_train = Y_train
        self.K = self.kernel(self.X_train, self.X_train) + sigma_y**2 * np.eye(len(self.X_train))
        
    def predict(self, X_s, sigma_y = 1):
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s)+ sigma_y**2 * np.eye(len(X_s))
        K_inv = inv(self.K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s,cov_s
    
    def RMS(self, datas, target):
        mean, var = self.predict(datas)
        error = np.sqrt(np.sum((mean-target)**2)/len(datas))
        return error
    
    def ard(self, lr):
        sqdist = np.sum(self.X_train**2, 1).reshape(-1, 1) + np.sum(self.X_train**2, 1) - 2 * np.dot(self.X_train, self.X_train.T)
        d_theta0 = np.exp(-0.5*self.thetas[1]*sqdist)
        d_theta1 = self.thetas[0] * np.exp(-0.5*self.thetas[1]*sqdist) * (-0.5*sqdist)
        d_theta2 = np.ones((len(self.X_train),len(self.X_train)))
        d_theta3 = self.X_train.dot(self.X_train.T)
        d_theta = [d_theta0,d_theta1,d_theta2,d_theta3]
        epoch = 0
        while True:
            update = np.zeros((4))
            flag = 0
            for i in range(4):
                update[i] = -0.5*np.trace(inv(self.K).dot(d_theta[i])) + 0.5*self.Y_train.T.dot(inv(self.K)).dot(d_theta[i]).dot(inv(self.K)).dot(self.Y_train)
                if np.absolute(update[i]) < 6.:
                    flag += 1
            self.thetas += lr*update
            self.K = self.kernel(self.X_train, self.X_train) + 1**2 * np.eye(len(self.X_train))
            epoch += 1
            if flag == 4:
                break

theta = [[0,0,0,1],[1,4,0,0],[1,4,0,5],[1,32,5,5]]
rms = []
for i in theta:
    gp = gaussian_process(i)
    line = np.linspace(0.,2.,100).reshape(100,1)
    gp.fit(x_train, t_train)
    mx, vx = gp.predict(line)
    vx = np.array([vx[j][j] for j in range(len(vx))])
    plt.plot(x_train, t_train,'bo')
    plt.plot(line, mx, linestyle = '-', color = 'red')
    plt.fill_between(line.reshape(-1), (mx.reshape(-1)-vx), (mx.reshape(-1)+vx), color = 'pink')
    plt.title('θ = [ '+str(i[0])+' , '+str(i[1])+' , '+str(i[2])+' , '+str(i[3])+' ]')
    plt.savefig(str(i)+'.png')
    plt.show()
    rms.append([gp.RMS(x_train, t_train),gp.RMS(x_test, t_test)])
import pandas as pd
rms = pd.DataFrame(rms)
rms.rename(columns={0:'train',1:'test'},inplace=True)
rms.rename(index={0:'[0,0,0,1]',1:'[1,4,0,0]',2:'[1,4,0,5]',3:'[1,32,5,5]'},inplace=True)
rms.to_excel('rms.xlsx')

theta = [3.,6.,4.,5.]
gp = gaussian_process(theta)
gp.fit(x_train, t_train)
gp.ard(0.001)
print(gp.RMS(x_train, t_train),gp.RMS(x_test, t_test))
line = np.linspace(0.,2.,100).reshape(100,1)
mx, vx = gp.predict(line)
vx = np.array([vx[j][j] for j in range(len(vx))])
plt.plot(x_train, t_train,'bo')
plt.plot(line, mx, linestyle = '-', color = 'red')
plt.fill_between(line.reshape(-1), (mx.reshape(-1)-vx), (mx.reshape(-1)+vx), color = 'pink')
plt.title('θ = [ '+str(round(gp.thetas[0],3))+' , '+str(round(gp.thetas[1],3))+' , '+str(round(gp.thetas[2],3))+' , '+str(round(gp.thetas[3],3))+' ]')
plt.show()
