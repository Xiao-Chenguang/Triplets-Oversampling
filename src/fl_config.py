import argparse

def get_parser():
    parser = argparse.ArgumentParser('Federated learning', description='Federate leanring algorithm in class imbalance scenario')

    # 1. Federation properties
    parser.add_argument('--task_name', type=str, default='', help='task name to use for results folder after scenario')
    parser.add_argument('--num_clients', type=int, default=100, help='total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=10, help='number of active client every iteration')
    parser.add_argument('--global_epochs', type=int, default=1000, help='number of global iteration')
    parser.add_argument('--eval_frequency', type=int, default=1, help='global evaluation frequency')

    # 2. local training property
    parser.add_argument('--local_epoch', type=int, default=2, help='local iteration per global round')
    parser.add_argument('--local_lr', type=float, default=1e-3, help='local learning rate in optimization')
    parser.add_argument('--local_momentum', type=float, default=0, help='local momentum in SGD optimization')

    # 3. model parameters
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to be learned')
    parser.add_argument('--batch_size', type=int, default=64, help='local training batch size')
    parser.add_argument('--device', type=str, default='cuda', help='device used')

    # 4. class imbalance parameters
    parser.add_argument('--ir', type=int, default=1, help='imbalance ratio (n_majority / n_minority)')

    # 5. multiprocessing parameters
    parser.add_argument('--jobid', type=int, default=0, help='job id for multiprocessing')

    # 6. logging parameters
    parser.add_argument('--log_level', type=str, default='INFO', help='logging level')

    return parser.parse_args()
    # used default value and avoid conflict in ipython
    # return parser.parse_args(args=[])