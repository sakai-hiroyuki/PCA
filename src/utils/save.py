import os
import numpy as np
import pandas as pd


def save(optimizer, dataset_name, name):
    if not os.path.isdir(f'results/history/{dataset_name}'):
        os.makedirs(f'results/history/{dataset_name}')
    if not os.path.isdir(f'results/ndarray/{dataset_name}'):
        os.makedirs(f'results/ndarray/{dataset_name}')
    
    df = pd.DataFrame(optimizer.history)
    df.to_csv(f'results/history/{dataset_name}/{name}.csv', header=False, index=False)
    np.save(f'results/ndarray/{dataset_name}/{name}', optimizer.xk)
