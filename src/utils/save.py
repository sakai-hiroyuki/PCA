import numpy as np
import pandas as pd


def save(optimizer, dataset, name):
    df = pd.DataFrame(optimizer.history)
    df.to_csv(f'results/history/{dataset}/{name}.csv', header=False, index=False)
    np.save(f'results/ndarray/{dataset}/{name}', optimizer.xk)
