import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RSGD(Optimizer):
    def __init__(self, lr: float=1e-3, beta: float=0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
    
    def update(self, M, xk, g, k):
        xk = M.retraction(xk, -self.lr * g)
        return xk
