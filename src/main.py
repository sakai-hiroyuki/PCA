from argparse import ArgumentParser

from optimizers import RSGD, RSVRG, RAdam, RAdaBound
from utils import create_loss, get_initial, save
from plot import plot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--components', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-2)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    components = args.components
    n_iter = args.n_iter
    lr = args.lr

    loss, data, _min = create_loss(dataset_name, components)
    N, n = data.shape[0], data.shape[1]

    optimizer_dict = {
        'SGD': RSGD(lr=0.05),
        'SVRG': RSVRG(lr=0.05, update_frequency=5*N),
        'Adam': RAdam(lr=0.05),
        'AdaBound': RAdaBound(lr=0.05)
    }

    x0 = get_initial(n, components)
    for name in optimizer_dict:
        print(name)
        optimizer = optimizer_dict[name]
        xk = optimizer.optimize(loss, data, components, n_iter=50*N, x0=x0)

        save(optimizer, dataset_name, name)
    
    plot(dataset_name, [key for key in optimizer_dict], _min)
