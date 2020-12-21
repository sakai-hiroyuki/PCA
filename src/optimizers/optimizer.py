import numpy as np

from autograd import grad
from time import time
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from manifolds import Stiefel


class Optimizer(object, metaclass=ABCMeta):
    def __init__(self):
        self.state = {}
    
    def optimize(self, loss, data, components, n_iter: int=20000, x0: np.ndarray=None):
        N = data.shape[0]
        n = data.shape[1]
        self.data = data
        M = Stiefel(components, n)
        if x0 is None:
            xk = np.linalg.qr(np.random.randn(n, r))[0]
        else:
            xk = x0.copy()
        
        self.logging(loss(xk))
        for k in tqdm(range(1, n_iter + 1)):
            xk = self.update(M, xk, k)
            
            if k % 100 == 0:
                self.logging(loss(xk))

        self.xk = xk

    @abstractmethod
    def update(self, M, xk, k) -> np.ndarray:
        raise NotImplementedError()

    def logging(self, v: float):
        if hasattr(self, 'history'):
            toc = time()
            self.history.append([v, toc - self.tic])
        else:
            self.tic = time()
            self.history = [[v, 0.]]
