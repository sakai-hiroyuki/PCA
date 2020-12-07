import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RAdam(Optimizer):
    def __init__(self, lr: float=1e-3, beta1: float=0.9, beta2: float=0.999, amsgrad: bool=False):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
    
    def update(self, M, xk, g, k):
        if not hasattr(self, 'params'):
            self.params = {}
            self.params['m'] = np.zeros_like(xk)
            self.params['tau'] = np.zeros_like(xk)
            self.params['v'] = 0.
            self.params['vhat'] = 0.

        self.params['m'] = self.beta1 * self.params['tau'] + (1 - self.beta1) * g
        if not self.amsgrad:
            self.params['mhat'] = self.params['m'] / (1 - self.beta1 ** (k + 1))
        else:
            self.params['mhat'] = self.params['m']
        self.params['v'] = self.beta2 * self.params['v'] + (1 - self.beta2) * np.trace(np.dot(g.T, g))
        if not self.amsgrad:
            self.params['vhat'] = self.params['v'] / (1 - self.beta2 ** (k + 1))
        else:
            self.params['vhat'] = max(self.params['vhat'], self.params['v'])
        
        xk = M.retraction(xk, -self.lr * self.params['mhat'] / (np.sqrt(self.params['vhat']) + 1e-8))
        self.params['tau'] = M.projection(xk, self.params['m'])
        
        return xk
