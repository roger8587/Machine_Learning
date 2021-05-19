import numpy as np
from numpy.linalg import inv, det, pinv
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if train_on_gpu else 'cpu')
device
img = Image.open('hw3_3.jpeg')
img = np.asarray(img, dtype='float')/255
img = torch.tensor(img).to(device)

class Kmeans(torch.nn.Module):
    def __init__(self, img, k):
        super(Kmeans, self).__init__()
        self.m, self.n, self.channel = img.size()
        self.img = img.view(-1, self.channel)
        self.k = k
        self.train_on_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.train_on_gpu else 'cpu')
        self.means = torch.rand(self.k, self.channel).to(self.device)
        self.r = torch.full((self.img.size()[0],), self.k + 1, dtype=torch.long).to(self.device)
        
    def fit(self, max_iter = 300):
        for i in range(max_iter):
            dist = torch.sum((self.img[:, None] - self.means) ** 2, axis=2)
            new_r = torch.argmin(dist, axis=1)
            if torch.equal(self.r, new_r):
                break
            else:
                self.r = new_r
            for j in range(self.k):
                data_k = self.img[torch.nonzero(torch.where(self.r == j,torch.full_like(self.r, 1),torch.full_like(self.r, 0)), as_tuple=True)]
                if len(data_k) == 0:
                    self.means[j] = torch.rand(self.channel)
                else:
                    self.means[j] = torch.mean(data_k, axis=0)
        
    def predict(self):
        new_data = torch.round(self.means[self.r]*255)
        if self.train_on_gpu:
            disp = Image.fromarray(new_data.view(self.m, self.n, self.channel).cpu().numpy().astype('uint8'))
        else:
            disp = Image.fromarray(new_data.view(self.m, self.n, self.channel).numpy().astype('uint8'))
        disp.show(title='k-means')
        disp.save('k-means(pytorch)_'+str(self.k)+'.png')
     
k = Kmeans(img,10).to(device)
k.fit()
k.predict()
def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
class GaussianMixture(torch.nn.Module):
    def __init__(self, img, k):
        super(GaussianMixture, self).__init__()
        self.m, self.n, self.channel = img.size()
        self.total_length = self.m*self.n
        self.img = img.view(-1, self.channel)
        self.k = k
        self.train_on_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.train_on_gpu else 'cpu')
        self.K_means = Kmeans(img,k)
        self.K_means.to(self.device)
        self.K_means.fit()
        self.rnk = self.K_means.r
        self.means = self.K_means.means
        
    def _init_parms(self):
        self.pi = torch.tensor([torch.nonzero(torch.where(self.rnk == i,torch.full_like(self.rnk, 1),torch.full_like(self.rnk, 0)), as_tuple=True)[0].size(0)/float(self.total_length) for i in range(self.k)])
        self.Cov = torch.cat([cov(self.img[torch.nonzero(torch.where(self.rnk == i,torch.full_like(self.rnk, 1),torch.full_like(self.rnk, 0)), as_tuple=True)]).unsqueeze(0) for i in range(self.k)],dim = 0)
        #self.Cov = torch.tensor([cov(self.img[torch.nonzero(torch.where(self.rnk == i,torch.full_like(self.rnk, 1),torch.full_like(self.rnk, 0)), as_tuple=True)]).detach().numpy() for i in range(self.k)])
        #self.psb = torch.exp(torch.tensor([MultivariateNormal(self.means[i],self.Cov[i]).log_prob(self.img).detach().numpy() for i in range(self.k)]).T)*self.pi
        self.psb = torch.exp(torch.cat([MultivariateNormal(self.means[i],self.Cov[i]).log_prob(self.img).unsqueeze(0) for i in range(self.k)],dim = 0).T)*self.pi
        print(self.psb.size())
    def fit(self, max_iter = 200, delta = 1e-3):
        likelihood = []
        self._init_parms()
        log_likelihood_old = self.log_likelihood().item()
        likelihood.append(log_likelihood_old)
        i = 1
        j = np.inf
        while i<= max_iter or j >= delta:
            self.E_step()
            self.M_step()
            log_likelihood = self.log_likelihood()
            i += 1
            j = log_likelihood - log_likelihood_old
            log_likelihood_old = log_likelihood
            likelihood.append(log_likelihood_old)
            
        plt.plot(range(len(likelihood)),likelihood)
        plt.title('GMM maximum likelihood curve')
        plt.xlabel('iterations')
        plt.ylabel('log p(x)')
        plt.show()
        
        
    def E_step(self):
        self.beta = self.psb/torch.sum(self.psb, axis=1).view(-1,1)
    
    def M_step(self):
        N = torch.sum(self.beta, axis=0)
        self.means = torch.sum(self.img[:, None] * self.beta[:, :, None], axis=0)/N[:, None]
        #self.Cov = torch.tensor([(((self.img - self.means[j]) * self.beta[:, j, None]).T.matmul(self.img - self.means[j])/N[j]).detach().numpy() for j in range(self.k)])        
        for j in range(self.k):
            self.Cov[j] = ((self.img - self.means[j]) * self.beta[:, j, None]).T.matmul(self.img - self.means[j])/N[j]
        self.pi = N/self.total_length
        #self.psb = torch.exp(torch.tensor([MultivariateNormal(self.means[i],self.Cov[i]).log_prob(self.img).detach().numpy() for i in range(self.k)]).T)*self.pi
        for j in range(self.k):
            try:
                self.psb[:, j] = torch.exp(MultivariateNormal(self.means[j],self.Cov[j]).log_prob(self.img))*self.pi[j]
            except :
                #self.means[j] = torch.rand(self.channel)
                temp = torch.rand(self.channel, self.channel)
                self.Cov[j] = temp.matmul(temp.T)
                self.psb[:, j] = torch.exp(MultivariateNormal(self.means[j],self.Cov[j]).log_prob(self.img))*self.pi[j]
    def log_likelihood(self):
        li = torch.tensor([MultivariateNormal(self.means[i],self.Cov[i]).log_prob(self.img).detach().numpy() for i in range(self.k)]).T + torch.log(self.pi)
        return torch.logsumexp(li, dim=1).sum()
    
    def predict(self):
        r = torch.argmax(self.psb, axis=1)
        new_data = torch.round(self.means[r]*255)
        disp2 = Image.fromarray(new_data.view(self.m, self.n, self.channel).numpy().astype('uint8'))
        disp2.show(title='GMM')
        disp2.save('GMM_'+str(self.k)+'.png')
gmm = GaussianMixture(img, 10)
gmm.to(device)
gmm.fit()
