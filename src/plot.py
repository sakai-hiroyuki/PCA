import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from load import get_mnist, get_digits
from utils import get_min, create_covariance_matrix


optimizers = ['SD1', 'AG1', 'AD1', 'AM1', 'AB1']
datasets = {'MNIST': get_mnist, 'digits': get_digits}


def plot(dataset, optimizers, _min: float):
    for name in optimizers:
        df = pd.read_csv(f'results/history/{dataset}/{name}.csv', header=None)[0]
        y = (df.values.flatten() - _min).flatten()
        x = range(len(y))

        plt.plot(x, y, label=name)
    
    xticks, _ = plt.xticks()
    xleft, xright = plt.xlim()
    plt.xticks(xticks, [f'{int(x)}' for x in 100 * xticks])
    plt.xlim(xleft, xright)
    
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Optimality gap')
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--components', type=int, default=10)

    args = parser.parse_args()
    dataset = args.dataset
    components = args.components

    data = datasets[dataset]()
    N = data.shape[0]

    C = np.load(f'data/{dataset}/C.npy')

    def loss(x):
        return (-np.trace(np.dot(x.T, np.dot(C, x))) + np.trace(C)) / N

    _min = get_min(loss, C, components)

    plot(dataset, optimizers, _min)
