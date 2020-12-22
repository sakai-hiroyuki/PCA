import numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RSGD(Optimizer):
    def __init__(self, lr=1e-3) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        self.lr = lr
        super(RSGD, self).__init__()
    
    def update(self, M, xk, k) -> np.ndarray:
        data = self.data
        N = data.shape[0]
        n = data.shape[1]

        state = self.state
        if len(self.state) == 0:
            pass
        
        # Caluculate stochastic gradient
        index = k % N
        z = data[index].reshape((n, 1))
        g = M.projection(xk, -2 * np.dot(np.dot(z, z.T), xk))
        
        xk = M.retraction(xk, -self.lr * g)
        return xk
