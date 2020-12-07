import os
import autograd.numpy as np
import pandas as pd

from argparse import ArgumentParser

from manifolds import Stiefel
from optimizers import RSGD, RAdam
from load import get_mnist, get_digits
from utils import get_min, create_covariance_matrix, save
from plot import plot


datasets = {'MNIST': get_mnist, 'digits': get_digits}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--components', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=10000)

    args = parser.parse_args()
    dataset = args.dataset
    components = args.components
    n_iter = args.n_iter

    data = datasets[dataset]()
    N, n = data.shape[0], data.shape[1]

    if not os.path.exists(f'data/{dataset}/C.npy'):
        create_covariance_matrix(data, f'data/{dataset}')
    C = np.load(f'data/{dataset}/C.npy')

    def loss(x):
        return (-np.trace(np.dot(x.T, np.dot(C, x))) + np.trace(C)) / N

    _min = get_min(loss, C, components)
    
    optimizer_dict = {
        'SD1': RSGD(lr=5e-3),
        'AD1': RAdam(lr=5e-3),
        'AM1': RAdam(lr=5e-3, amsgrad=True)
    }

    x0 = np.linalg.qr(np.random.randn(n, components))[0]
    for name in optimizer_dict:
        print(name)
        optimizer = optimizer_dict[name]
        xk = optimizer.optimize(loss, data, components, n_iter=n_iter, x0=x0)

        save(optimizer, dataset, name)
    
    plot(dataset, [key for key in optimizer_dict], _min)
