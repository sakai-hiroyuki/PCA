import numpy as np

from tqdm import tqdm

from optimizers import Optimizer
from manifolds import Stiefel


class RCGVR(Optimizer):
    def __init__(self, lr=1e-3, update_frequency: int=2000) -> None:
        if not 0. < lr:
            raise ValueError(f'Invaild learning rate: {lr}')
        self.lr = lr
        self.update_frequency = update_frequency
        super(RCGVR, self).__init__()

    def update(self, M, xk, k) -> np.ndarray:
        data = self.data
        N = data.shape[0]
        n = data.shape[1]

        state = self.state
        if len(self.state) == 0:
            state['iter_count'] = 0
            state['store_xk'] = np.zeros_like(xk)
            state['store_fullgradient'] = np.zeros_like(xk)
            state['current_g'] = np.zeros_like(xk)
            state['previous_g'] = np.zeros_like(xk)
            state['search_dir'] = np.zeros_like(xk)
        
        if state['iter_count'] == 0:
            full_gradient = np.zeros_like(xk)
            for index in range(N):
                z = data[index].reshape((n, 1))
                full_gradient += M.projection(xk, -2 * np.dot(np.dot(z, z.T), xk))
            state['store_xk'] = xk
            state['store_full_gradient'] = full_gradient / N
            
            if k == 1:
                state['current_g'] = full_gradient
                state['search_dir'] = -full_gradient
        
        store_xk = state['store_xk']
        store_full_gradient = state['store_full_gradient']
        
        xk = M.retraction(xk, self.lr * state['search_dir'])

        # Caluculate stochastic gradient
        index = np.random.randint(0, N)
        z = data[index].reshape((n, 1))
        g = M.projection(xk, -2 * np.dot(np.dot(z, z.T), xk))
        g_back = M.projection(store_xk, -2 * np.dot(np.dot(z, z.T), store_xk))
        
        state['previous_g'] = state['current_g']
        state['current_g'] = g - M.projection(xk, g_back - store_full_gradient)

        previous_g = state['previous_g']
        current_g = state['current_g']

        numer = np.trace(np.dot(current_g, current_g.T))
        denom = np.trace(np.dot(previous_g, previous_g.T))
        beta = numer / denom

        state['search_dir'] = -state['current_g'] + beta * M.projection(xk, state['search_dir']) / np.sqrt(k)

        state['iter_count'] += 1
        if state['iter_count'] > self.update_frequency:
            state['iter_count'] = 0
        
        return xk