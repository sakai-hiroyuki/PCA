import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RAdaGrad(Optimizer):
    def __init__(self, lr: float=1e-3):
        super().__init__()
        self.lr = lr
    
    def update(self, M, xk, g, k):
        if not hasattr(self, 'params'):
            self.params = {}
            self.params['v'] = 0.

        self.params['v'] += np.trace(np.dot(g.T, g))
        
        xk = M.retraction(xk, -self.lr * g / (np.sqrt(self.params['v']) + 1e-8))
        
        return xk
