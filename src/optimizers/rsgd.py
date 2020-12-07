import autograd.numpy as np

from tqdm import tqdm
from autograd import grad

from optimizers import Optimizer
from manifolds import Stiefel


class RSGD(Optimizer):
    def __init__(self, lr: float=1e-3, beta: float=0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
    
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
                
            xk = M.retraction(xk, -self.lr * g)
            
            if k % 100 == 0:
                l = loss(xk)
                self.logging(l)

        self.xk = xk