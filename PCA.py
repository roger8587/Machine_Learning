import numpy as np

class PCA(object):
    def __init__(self,data,ncomponent):
        self.data = data
        self.n = ncomponent
        self.N = np.size(self.data,0)
    def zeroMean(self): #按列求均值，即求各個特徵的均值
        self.meanVal = np.mean(self.data,axis = 0)
        newData = self.data - self.meanVal       
        return newData
    def get_CovarianceMatrix(self,X): #協方差矩陣
        Cov = np.dot(np.transpose(X),X)/self.N        
        return Cov
    def DimensionReduction(self):
        X1 = self.zeroMean()
        Cov = self.get_CovarianceMatrix(X1)
        eigVects, eigVals, U = np.linalg.svd(Cov)
        eigValIndice = np.argsort(eigVals) #對特徵值從小到大排序
        n_eigValIndice = eigValIndice[-1:-(self.n+1):-1] 
        n_eigVect = eigVects[:, n_eigValIndice] #最大的n個特徵值對應的特徵向量
        lowDX = np.dot(X1, n_eigVect) #低維特徵空間的資料
        reconX = np.dot(lowDX, n_eigVect.T) + self.meanVal #重構資料
        return lowDX, reconX
