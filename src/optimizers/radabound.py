import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RAdaBound(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, eps=1e-8, amsbound=False) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        if not 0.0 < eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.final_lr = final_lr
        self.amsbound = amsbound
        super(RAdaBound, self).__init__()
    
    def update(self, M, xk, g, k) -> np.ndarray:
        if not hasattr(self, 'state'):
            self.state = {}
            self.state['m'] = np.zeros_like(xk)
            self.state['tau'] = np.zeros_like(xk)
            self.state['v'] = 0.
            self.state['vhat'] = 0.

        self.state['m'] = self.betas[0] * self.state['tau'] + (1 - self.betas[0]) * g
        self.state['v'] = self.betas[1] * self.state['v'] + (1 - self.betas[1]) * np.trace(np.dot(g.T, g))
        if self.amsbound:
            self.state['vhat'] = max(self.state['vhat'], self.state['v'])
        else:
            self.state['vhat'] = self.state['v']

        lower = self.final_lr * (1 - 1 / ((1 - self.betas[1]) * k + 1))
        upper = self.final_lr * (1 + 1 / ((1 - self.betas[1]) * k))

        clipped = np.clip(self.lr / (np.sqrt(self.state['vhat']) + self.eps), lower, upper)
        
        xk = M.retraction(xk, -clipped * self.state['m'])
        self.state['tau'] = M.projection(xk, self.state['m'])
        
        return xk
