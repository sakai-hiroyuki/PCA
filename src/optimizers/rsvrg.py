import numpy as np

from tqdm import tqdm

from optimizers import Optimizer
from manifolds import Stiefel


class RSVRG(Optimizer):
    def __init__(self, lr=1e-3, update_frequency: int=2000) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        self.lr = lr
        self.update_frequency = update_frequency
        super(RSVRG, self).__init__()

    def update(self, M, xk, k) -> np.ndarray:
        data = self.data
        N = data.shape[0]
        n = data.shape[1]

        state = self.state
        if len(self.state) == 0:
            state['iter_count'] = 0
            state['store_xk'] = xk
            state['store_fullgradient'] = np.zeros_like(xk)
        
        if state['iter_count'] == 0:
            full_gradient = np.zeros_like(xk)
            for index in range(N):
                z = data[index].reshape((n, 1))
                full_gradient += M.projection(xk, -2 * np.dot(np.dot(z, z.T), xk))
            state['store_xk'] = xk
            state['store_full_gradient'] = full_gradient / N
        
        store_xk = state['store_xk']
        store_full_gradient = state['store_full_gradient']

        # Caluculate stochastic gradient
        # index = np.random.randint(0, N)
        index = k % N
        z = data[index].reshape((n, 1))
        g = M.projection(xk, -2 * np.dot(np.dot(z, z.T), xk))

        g_back = M.projection(store_xk, -2 * np.dot(np.dot(z, z.T), store_xk))
        
        search_dir = -g + M.projection(xk, g_back - store_full_gradient)

        state['iter_count'] += 1
        if state['iter_count'] > self.update_frequency:
            state['iter_count'] = 0
        
        xk = M.retraction(xk, self.lr * search_dir)
        return xk
