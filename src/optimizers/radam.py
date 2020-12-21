import numpy as np

from optimizers import Optimizer
from manifolds import Stiefel


class RAdam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad: bool=False) -> None:
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
        self.amsgrad = amsgrad
        super(RAdam, self).__init__()
    
    def update(self, M, xk, g, k) -> np.ndarray:
        amsgrad = self.amsgrad
        state = self.state
        if len(state) == 0:
            # Exponential moving average of gradient values
            state['exp_avg'] = np.zeros_like(xk)
            # Exponential moving average of squared norm of gradient values
            state['exp_avg_sq'] = 0.
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. norm of grad. values
                state['max_exp_avg_sq'] = 0.
        
        beta1, beta2 = self.betas
        
        state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * g
        state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * np.trace(np.dot(g.T, g))
        if amsgrad:
            state['max_exp_avg_sq'] = max(state['exp_avg_sq'], state['max_exp_avg_sq'])
            denom = np.sqrt(state['max_exp_avg_sq']) + self.eps
            step_size = self.lr
        else:
            bias_correction1 = 1 - beta1 ** k
            bias_correction2 = 1 - beta2 ** k
            denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + self.eps
            step_size = self.lr / bias_correction1
        
        step_size /= denom
        
        xk = M.retraction(xk, -step_size * state['exp_avg'])
        state['exp_avg'] = M.projection(xk, state['exp_avg'])
        
        return xk
