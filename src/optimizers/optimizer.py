import autograd.numpy as np

from autograd import grad
from time import time
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from manifolds import Stiefel


class Optimizer(object, metaclass=ABCMeta):
    def __init__(self):
        pass
    
    def optimize(self, loss, data, components, n_iter: int=20000, x0: np.ndarray=None):
        N = data.shape[0]
        n = data.shape[1]
        M = Stiefel(components, n)
        if x0 is None:
            xk = np.linalg.qr(np.random.randn(n, r))[0]
        else:
            xk = x0.copy()

        for k in tqdm(range(n_iter)):
            index = np.random.randint(0, N)
            z = data[index]
        
            def loss_i(x):
                return - np.dot(np.dot(z, x), np.dot(x.T, z))
            g = M.projection(xk, grad(loss_i)(xk))
            
            xk = self.update(M, xk, g, k)
            
            if k % 100 == 0:
                self.logging(loss(xk))

        self.xk = xk

    @abstractmethod
    def update(self, M, xk, g, k):
        raise NotImplementedError()

    def logging(self, v: float):
        if hasattr(self, 'history'):
            toc = time()
            self.history.append([v, toc - self.tic])
        else:
            self.tic = time()
            self.history = [[v, 0.]]
