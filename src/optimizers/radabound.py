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
        amsbound = self.amsbound
        state = self.state
        if len(state) == 0:
            # Exponential moving average of gradient values
            state['exp_avg'] = np.zeros_like(xk)
            # Exponential moving average of squared norm of gradient values
            state['exp_avg_sq'] = 0.
            if amsbound:
                # Maintains max of all exp. moving avg. of sq. norm of grad. values
                state['max_exp_avg_sq'] = 0.
        
        beta1, beta2 = self.betas
        
        state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * g
        state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * np.trace(np.dot(g.T, g))
        if amsbound:
            state['max_exp_avg_sq'] = max(state['exp_avg_sq'], state['max_exp_avg_sq'])
            denom = np.sqrt(state['max_exp_avg_sq']) + self.eps
        else:
            denom = np.sqrt(state['exp_avg_sq']) + self.eps

        final_lr = self.final_lr
        lower_bound = final_lr * (1 - 1 / (beta2 * k + 1))
        upper_bound = final_lr * (1 + 1 / (beta2 * k))

        step_size = self.lr / denom
        step_size = np.clip(step_size, lower_bound, upper_bound)
        
        xk = M.retraction(xk, -step_size * state['exp_avg'])
        state['exp_avg'] = M.projection(xk, state['exp_avg'])
        
        return xk
