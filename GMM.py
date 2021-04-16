import numpy as np
from numpy.linalg import inv, det, pinv
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('hw3_3.jpeg')
img = np.asarray(img, dtype='float')/255

class Kmeans():
    def __init__(self, img, k):
        self.m, self.n, self.channel = img.shape
        self.img = np.reshape(img, (-1, self.channel))
        self.k = k
        self.means = np.random.rand(self.k, self.channel)
        self.r = np.full([self.img.shape[0]], self.k + 1)
        
    def fit(self, max_iter = 300):
        for i in range(max_iter):
            dist = np.sum((self.img[:, None] - self.means) ** 2, axis=2)
            new_r = np.argmin(dist, axis=1)
            if np.array_equal(self.r, new_r):
                break
            else:
                self.r = new_r
            for j in range(self.k):
                data_k = self.img[np.where(self.r == j)]
                if len(data_k) == 0:
                    self.means[j] = np.random.rand(self.channel)
                else:
                    self.means[j] = np.mean(data_k, axis=0)
        new_data = np.round(self.means[self.r]*255)
        disp = Image.fromarray(new_data.reshape(self.m, self.n, self.channel).astype('uint8'))
        return self.means
        #disp.show(title='k-means')
        #disp.save('k-means_'+str(k)+'.png')
    def get_item(self):
        pi = np.array([len(np.where(self.r == i)[0])/float(self.img.shape[0]) for i in range(self.k)])
        cov = np.array([np.cov(self.img[np.where(self.r == i)].T) for i in range(self.k)])
        return self.means, cov, pi
        
k = Kmeans(img,5)
k.fit()
