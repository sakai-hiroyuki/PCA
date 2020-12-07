import autograd.numpy as np

from autograd import grad
from tqdm import tqdm

from optimizers import Optimizer
from manifolds import Stiefel


class RAdam(Optimizer):
    def __init__(self, lr: float=1e-3, beta1: float=0.9, beta2: float=0.999, amsgrad: bool=False):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
    
    def optimize(self, loss, data, components, n_iter: int=20000, x0: np.ndarray=None):
        N = data.shape[0]
        n = data.shape[1]
        M = Stiefel(components, n)
        if x0 is None:
            xk = np.linalg.qr(np.random.randn(n, components))[0]
        else:
            xk = x0.copy()

        m = np.zeros_like(xk)
        tau = np.zeros_like(xk)
        v = 0.
        vhat = 0.
        for k in tqdm(range(n_iter)):
            index = np.random.randint(0, N)
            z = data[index]
        
            def loss_i(x):
                return - np.dot(np.dot(z, x), np.dot(x.T, z))
            g = M.projection(xk, grad(loss_i)(xk))
            
            m = self.beta1 * tau + (1 - self.beta1) * g
            if not self.amsgrad:
                mhat = m / (1 - self.beta1 ** (k + 1))
            else:
                mhat = m
            v = self.beta2 * v + (1 - self.beta2) * np.trace(np.dot(g.T, g))
            if not self.amsgrad:
                vhat = v / (1 - self.beta2 ** (k + 1))
            else:
                vhat = max(vhat, v)
            
            xk = M.retraction(xk, -self.lr * mhat / (np.sqrt(vhat) + 1e-8))
            tau = M.projection(xk, m)
            
            if k % 100 == 0:
                l = loss(xk)
                self.logging(l)

        self.xk = xk