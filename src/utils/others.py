import os
import numpy as np
from scipy.linalg import eig
from tqdm import tqdm

from load import get_mnist, get_digits, get_iris


datasets = {
    'MNIST': get_mnist,
    'digits': get_digits,
    'iris': get_iris
}


def create_loss(dataset_name: str, components: int):
    data = datasets[dataset_name]()
    N, n = data.shape
    C = _load_covariance_matrix(dataset_name, use_tqdm=True)

    def loss(x):
        return (-np.trace(np.dot(x.T, np.dot(C, x))) + np.trace(C)) / N
    
    _min = _get_min(loss, C, components)
    
    return loss, data, _min


def _get_min(loss, C, components):
    _opt = _get_optim(C, components)
    return loss(_opt)


def _get_optim(C, components):
    eigen_value, eigen_vector = eig(C)
    eigen_id = np.argsort(eigen_value)[::-1]
    eigen_vector = eigen_vector[:,eigen_id]
    return eigen_vector[:, 0:components]


def _load_covariance_matrix(dataset_name, use_tqdm=True):
    if not os.path.isdir(f'./data/{dataset_name}'):
        os.makedirs(f'./data/{dataset_name}')
    if os.path.isfile(f'./data/{dataset_name}/cvm.npy'):
        print('The covariance matrix is already exists.')
    else:
        _create_covariance_matrix(dataset_name, use_tqdm)
    
    return np.load(f'./data/{dataset_name}/cvm.npy')


def _create_covariance_matrix(dataset_name, use_tqdm=True):
    print('Creating the covariance matrix ...')
    data = datasets[dataset_name]()
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

    np.save(f'./data/{dataset_name}/cvm', C)


def get_initial(n, components):
    # return np.linalg.qr(np.random.randn(n, components))[0]
    return np.linalg.qr(np.ones((n, components)))[0]
