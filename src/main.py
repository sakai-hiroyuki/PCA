from argparse import ArgumentParser

from manifolds import Stiefel
from optimizers import RSGD, RAdam, RAdaGrad, RAdaBound
from load import get_mnist, get_digits, get_iris
from utils import create_loss, load_covariance_matrix, get_initial, save
from plot import plot


datasets = {
    'MNIST': get_mnist,
    'digits': get_digits,
    'iris': get_iris
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--components', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-3)

    args = parser.parse_args()
    dataset = args.dataset
    components = args.components
    n_iter = args.n_iter
    lr = args.lr

    optimizer_dict = {
        'SD1': RSGD(lr=lr),
        'AG1': RAdaGrad(lr=lr),
        'AD1': RAdam(lr=lr),
        'AM1': RAdam(lr=lr, amsgrad=True),
        'ADB1': RAdaBound(lr=lr),
        'AMB1': RAdaBound(lr=lr, amsbound=True)
    }

    data = datasets[dataset]()
    N, n = data.shape[0], data.shape[1]

    C = load_covariance_matrix(data, dataset)
    loss, _min = create_loss(C, components, N)

    x0 = get_initial(n, components)
    for name in optimizer_dict:
        print(name)
        optimizer = optimizer_dict[name]
        xk = optimizer.optimize(loss, data, components, n_iter=n_iter, x0=x0)

        save(optimizer, dataset, name)
    
    plot(dataset, [key for key in optimizer_dict], _min)
