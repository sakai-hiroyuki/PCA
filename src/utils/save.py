import os
import numpy as np
import pandas as pd


def save(optimizer, dataset, name):
    if not os.path.isdir(f'results/history/{dataset}'):
        os.makedirs(f'results/history/{dataset}')
    if not os.path.isdir(f'results/ndarray/{dataset}'):
        os.makedirs(f'results/ndarray/{dataset}')
    
    df = pd.DataFrame(optimizer.history)
    df.to_csv(f'results/history/{dataset}/{name}.csv', header=False, index=False)
    np.save(f'results/ndarray/{dataset}/{name}', optimizer.xk)
