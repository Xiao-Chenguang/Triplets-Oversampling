# import packages
import torch
import numpy as np
from itertools import product
from src.fl_fedAvg import FedAvg
from src.fl_alexNet import AlexNet
from src.fl_config import get_parser
from src.cl_data import load_vision_data, resampling
from torch.utils.data import DataLoader, TensorDataset


def run_nn(dataset, os="nonsampling", ir=1, seed=0, task_name=''):
    args = get_parser()

    # ========== define the parameters ==========
    args.task_name = task_name
    args.dataset = dataset
    # nonmsapling oversampling smote blsmote adasyn triplet
    args.os = os
    args.seed = seed
    args.ir = ir

    # the rest parameters keeps same as the default value
    args.global_epochs = 1000
    args.local_epoch = 2

    args.num_clients = 100
    args.clients_per_round = 10

    args.local_lr = 0.001
    args.local_momentum = 0.9

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ================== prepare data ==================
    channel = 1 if args.dataset == 'mnist' else 3
    dim = 28 if args.dataset == 'mnist' else 32

    x_train, x_test, y_train, y_test = load_vision_data(args.dataset)
    # convert to numpy
    if type(x_train) == torch.Tensor:
        x_train, x_test = x_train.numpy(), x_test.numpy()
        y_train, y_test = y_train.numpy(), y_test.numpy()
    else:
        y_train, y_test = np.array(y_train), np.array(y_test)
    # get the original shape
    n, d = x_train.shape[0], x_train.shape[1:]
    x_train = x_train.reshape(n, -1)
    x_test = x_test.reshape(-1, channel, dim, dim)
    y_test = y_test.reshape(-1, 1)
    # select and set the minority class to 1
    cmin = np.random.randint(10)
    y_train = (y_train == cmin).astype('int')
    y_test = (y_test == cmin).astype('int')
    x_train = x_train / 255.
    x_test = x_test / 255.
    rid = np.random.permutation(y_train.shape[0])
    x_train, y_train = x_train[rid], y_train[rid]

    # split the data into clients
    n_min = int((y_train == 1).sum() / args.ir / args.num_clients)
    n_maj = int((y_train == 0).sum() / args.num_clients)
    min_idx = np.where(y_train == 1)[0]
    maj_idx = np.where(y_train == 0)[0]
    cid = [np.concatenate([min_idx[i*n_min:(i+1)*n_min],
                        maj_idx[i * n_maj:(i+1)*n_maj]]) for i in range(args.num_clients)]
    fed_x = [x_train[cid[i]] for i in range(args.num_clients)]
    fed_y = [y_train[cid[i]] for i in range(args.num_clients)]

    # sample the test data
    res_fed_x = []
    res_fed_y = []
    for cx, cy in zip(fed_x, fed_y):
        res_x, res_y = x_train, y_train = resampling(cx, cy, sampling=args.os, len_lim=True, random=True)
        res_fed_x.append(res_x.reshape(-1, channel, dim, dim))
        res_fed_y.append(res_y.reshape(-1, 1))
    # res_fed_x = np.concatenate(res_fed_x, axis=0)
    # res_fed_y = np.concatenate(res_fed_y, axis=0)
    # print(res_fed_x.shape, res_fed_y.shape)
    # convert to data loader
    fed_ds = [TensorDataset(torch.Tensor(res_fed_x[i]), torch.Tensor(res_fed_y[i])) for i in range(args.num_clients)]
    fed_dl = [DataLoader(fed_ds[i], batch_size=args.batch_size, shuffle=True) for i in range(args.num_clients)]
    test_ds = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 8)


    # ====================== get the model ======================

    model = AlexNet(channel, dim, dim, 1).to(args.device)


    # ====================== train the model ======================
    fed = FedAvg(fed_dl, test_dl, model, args)
    fed.train()


dss =  ['mnist', 'cifar10']
oss =  ['nonsampling', 'oversampling', 'smote', 'blsmote', 'adasyn', 'triplets_m']
irs= [1, 2, 4, 8]
seeds = range(30)

for job in product(dss, oss, irs, seeds):
    run_nn(dataset=job[0], os=job[1], ir=job[2], seed=job[3], task_name='fl')
