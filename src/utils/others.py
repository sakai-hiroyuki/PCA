import os
import numpy as np
from scipy.linalg import eig
from tqdm import tqdm


def get_min(loss, C, components):
    _opt = _get_optim(C, components)
    return loss(_opt)


def _get_optim(C, components):
    eigen_value, eigen_vector = eig(C)
    eigen_id = np.argsort(eigen_value)[::-1]
    eigen_vector = eigen_vector[:,eigen_id]
    return eigen_vector[:, 0:components]


def create_covariance_matrix(data, dir: str, use_tqdm=True):
    print('creating covariance matrix ...')
    N, n = data.shape
    C = np.zeros((n, n))
    if use_tqdm:
        for index in tqdm(range(N)):
            z = np.reshape(data[index], (n, 1))
            C += np.dot(z, z.T)
    else:
        for index in range(N):
            z = np.reshape(data[index], (n, 1))
            C += np.dot(z, z.T)
    np.save(os.path.join(dir, 'C'), C)
