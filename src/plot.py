import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import create_loss


def plot(dataset_name, optimizers, _min: float):
    for name in optimizers:
        df = pd.read_csv(f'results/history/{dataset_name}/{name}.csv', header=None)[0]
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
    pass
