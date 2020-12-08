import autograd.numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RSGD(Optimizer):
    def __init__(self, lr=1e-3) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        self.lr = lr
        super(RSGD, self).__init__()
    
    def update(self, M, xk, g, k) -> np.ndarray:
        if not hasattr(self, 'state'):
            self.state = {}

        xk = M.retraction(xk, -self.lr * g)
        return xk
