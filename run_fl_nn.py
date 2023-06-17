# import packages
import torch
from itertools import product
from src.fl_fedAvg import FedAvg
from src.fl_alexNet import AlexNet
from src.fl_config import get_parser
from src.fl_data import get_fed_dataset
from torch.utils.data import DataLoader


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
    channel = 1 if args.dataset == 'femnist' else 3
    dim = 28 if args.dataset == 'femnist' else 32
    fed_ds, test_ds = get_fed_dataset(args, channel, dim)
    fed_dl = [DataLoader(fed_ds[i], batch_size=args.batch_size, shuffle=True) for i in range(args.num_clients)]
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
