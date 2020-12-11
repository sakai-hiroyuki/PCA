import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RAdaGrad(Optimizer):
    def __init__(self, lr=1e-3, eps=1e-8) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        if not 0.0 < eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        self.lr = lr
        self.eps = eps
        super(RAdaGrad, self).__init__()
    
    def update(self, M, xk, g, k) -> np.ndarray:
        if len(self.state) == 0:
            self.state['v'] = 0.

        self.state['v'] += np.trace(np.dot(g.T, g))
        
        xk = M.retraction(xk, -self.lr * g / (np.sqrt(self.state['v']) + self.eps))
        
        return xk
