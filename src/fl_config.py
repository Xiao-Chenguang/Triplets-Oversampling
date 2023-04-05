import argparse

def get_parser():
    parser = argparse.ArgumentParser('Federated learning', description='Federate leanring algorithm in class imbalance scenario')

    # 1. Federation properties
    parser.add_argument('--num_clients', type=int, default=100, help='total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=10, help='number of active client every iteration')
    parser.add_argument('--global_epochs', type=int, default=1000, help='number of global iteration')
    parser.add_argument('--device', type=str, default='cuda', help='device used')
    parser.add_argument('--fed_alg', type=str, default='FEDAVG', choices=['FEDAVG', 'DMFL', 'MFL', 'AMFL', 'FEDCM'], help='federated learning algorithm')
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet18'], help='model used')

    # model parameters
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to be learned')
    parser.add_argument('--ds_path', type=str, default='/bask/homes/c/cxx075/Chenguang/datasets', help='root path to datasets')
    parser.add_argument('--batch_size', type=int, default=64, help='local training batch size')
    parser.add_argument('--drop_last', action='store_true', help='drop last batch if not enough data')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for client dataloader')
    parser.add_argument('--algorithm', type=str, default='nonsampling', help='sampling algorithm used in FL training')

    # evaluation parameters
    parser.add_argument('--eval_frequency', type=int, default=1, help='global evaluation frequency')
    parser.add_argument('--test_batch', type=int, default=512, help='test batch size')

    # 2. local training property
    parser.add_argument('--loss_fn', type=str, default='CELoss', help='loss function used in local training')
    parser.add_argument('--local_epoch', type=int, default=2, help='local iteration per global round')
    parser.add_argument('--local_lr', type=float, default=1e-3, help='local learning rate in optimization')
    parser.add_argument('--lr_decay', type=float, default=1, help='local learning rate decay')
    parser.add_argument('--global_lr', type=float, default=1, help='global learning rate in optimization')
    parser.add_argument('--global_momentum', type=float, default=0, help='momentum in global aggregation')
    parser.add_argument('--local_momentum', type=float, default=0, help='local momentum in SGD optimization')
    parser.add_argument('--local_dampening', action='store_true', help='local dampening in SGD optimization')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimization')
    parser.add_argument('--scenario', type=int, default=2, help='class imbalance scenario')
    parser.add_argument('--agg_adam', action='store_true', help='use aggregated adam to initial learners')
    parser.add_argument('--adam_moment1', type=float, default=0.9, help='moment1 for adam')
    parser.add_argument('--adam_moment2', type=float, default=0.999, help='moment2 for adam')
    parser.add_argument('--optim_alg', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta'], help='local optimizer')

    # class imbalance scenario 2 parameters
    parser.add_argument('--ir', type=int, default=1, help='imbalance ratio (n_majority / n_minority)')
    parser.add_argument('--num_minclass', type=int, default=0, help='number of minority classes')
    parser.add_argument('--client_class', type=int, default=10, help='number of classes in every clients')

    parser.add_argument('--task_name', type=str, default='', help='task name to use for results folder after scenario')
    parser.add_argument('--task_id', type=int, help='task id of this job sequence')

    parser.add_argument('--dir_alpha', type=float, default=0.1, help='alpha in dirichlet distribution')

    parser.add_argument('--rescale', action='store_true', help='rescale the initial momentum in local training')

    # parameter for FedSof
    parser.add_argument('--soft_alpha', type=float, default=0.1, help='factor for soft loss in FedSof')

    # parameter for FedSofSch
    parser.add_argument('--soft_decay', type=str, default='linear', choices=['linear', 'exp', 'invprop'], help='decay method for soft alpha in FedSofSch')
    parser.add_argument('--soft_decay_rate', type=int, default=0, help='decay factor for soft alpha in FedSofSch')

    # parameter for DMFLSCH
    parser.add_argument('--mom_decay', type=str, default='linear', choices=['linear', 'exp', 'invprop'], help='decay method for momentum in dmflsch')
    parser.add_argument('--mom_decay_rate', type=int, default=0, help='decay factor for momentum in dfmlsch')

    # parameter for BalanceFL
    parser.add_argument('--temperature', type=float, default=2, help='tempture for knowledge distillation')
    parser.add_argument('--soft', type=float, default=0.1, help='soft loss factor')

    # parameter for FedAvg_bal_warmup
    parser.add_argument('--warmup_rounds', type=int, default=0, help='warmup iterations for FedAvg_bal_warmup')

    # parameter for minmaj_sampling
    parser.add_argument('--minmaj_alpha', type=float, default=0.7, help='alpha for FedAvg_minmaj')

    # parameter for ClAug
    parser.add_argument('--claug_rate', type=int, default=1, help='ClAug all classes to claug_rate * max(class size)')

    # parameter for undaug_sampling
    parser.add_argument('--undaug_alpha', type=float, default=1, help='alpha for FedAvg_undaug')

    # parameter for mixup
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='alpha for mixup')

    return parser.parse_args()
    # used default value and avoid conflict in ipython
    # return parser.parse_args(args=[])