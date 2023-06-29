# import packages
import torch
from itertools import product
from src.fl_fedAvg import FedAvg
from src.fl_alexNet import AlexNet
from src.fl_config import get_parser
from src.fl_data import get_fed_dataset
from torch.utils.data import DataLoader
import logging


def run_nn(args, task_name=''):
    log_name = '_'.join([task_name, args.dataset, args.os,
                         str(args.ir), str(args.seed)])

    # clear log configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # set up new log configuration
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.FileHandler('log/' + log_name + ".log"),
            logging.StreamHandler()
        ]
    )

    # mute the PIL logging
    logging.getLogger('PIL').level = logging.WARNING
    logger = logging.getLogger(__name__)
    logger.info('start prepare the job')

    torch.manual_seed(args.seed)

    # =================== define the parameters ===================
    args.task_name = task_name

    # the rest parameters keeps same as the default value
    args.global_epochs = 1000
    args.eval_frequency = 10
    args.local_epoch = 2

    args.num_clients = 100
    args.clients_per_round = 10

    args.local_lr = 0.001
    args.local_momentum = 0.9

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ======================= prepare data ========================
    channel = 1 if args.dataset == 'femnist' else 3
    dim = 28 if args.dataset == 'femnist' else 32
    fed_ds, test_ds = get_fed_dataset(args, channel, dim)
    fed_dl = [DataLoader(fed_ds[i], batch_size=args.batch_size,
                         shuffle=True, num_workers=args.train_wks)
              for i in range(args.num_clients)]
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 8,
                         shuffle=False, num_workers=args.test_wks)

    # ======================= get the model =======================
    model = AlexNet(channel, dim, dim, 1).to(args.device)

    # ====================== train the model ======================
    fed = FedAvg(fed_dl, test_dl, model, args)
    fed.train()


dss = ['mnist', 'cifar10']
oss = ['nonsampling', 'oversampling', 'smote',
       'blsmote', 'adasyn', 'triplets_m']
irs = [1, 2, 4, 8]
seeds = range(30)

args = get_parser()

for job in product(dss, oss, irs, seeds):
    args.dataset, args.os, args.ir, args.seed = job
    run_nn(args, task_name='fl')
