import logging
import time
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.cl_data import load_vision_data, resampling


def np_data(h5wtr: h5py.Group, cmin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # scale data from 0-255 to 0-1
    x = h5py.Dataset(h5wtr["images"].id)[:] / 255.0
    # convert label from 30-39 to 0-9
    y = h5py.Dataset(h5wtr["labels"].id)[:]
    # convert to binary-class
    y = np.isin(y, cmin).astype(int)  # or float ?
    return x, y


def group_writers(h5file, writers, cmin):
    client_x = []
    client_y = []
    for writer in writers:
        x, y = np_data(h5file[writer], cmin)

        client_x.append(x)
        client_y.append(y)
    # combine the [group] writers into one dataset
    client_x = np.concatenate(client_x, axis=0)
    client_y = np.concatenate(client_y, axis=0)

    return client_x, client_y


def load_client(i, h5file, writers, cmin, ir, group, logger, os, channel, dim):
    logger.debug(f"load the {i}-th client")
    client_writers = writers[i * group : (i + 1) * group]
    client_x, client_y = group_writers(h5file, client_writers, cmin)

    # simulate the imbalanced dataset
    pos_id = np.where(client_y == 1)[0]
    neg_id = np.where(client_y == 0)[0]
    np.random.shuffle(pos_id)
    pos_id = pos_id[: int(len(pos_id) / ir)]
    sampled_id = np.concatenate((pos_id, neg_id), axis=0)

    client_x = client_x[sampled_id]
    client_y = client_y[sampled_id]
    client_x = client_x.reshape(client_x.shape[0], -1)

    client_x, client_y = resampling(
        client_x, client_y, sampling=os, len_lim=True, random=True
    )
    return client_x.reshape(-1, channel, dim, dim), client_y.reshape(-1)


def get_fed_dataset(args, channel, dim):
    np.random.seed(args.seed)
    logger = logging.getLogger(__name__)
    logger.info("start prepare the Fed datasets")
    cmin = np.random.choice(10, 5, replace=False)
    # cmin = np.random.randint(10)
    if args.dataset == "femnist":
        group = args.group_size  # allow ir of 8
        logger.info(
            f"load the femnist dataset with {args.num_clients} clients of group {group}"
        )
        root_path = "datasets/femnist/write_digits.hdf5"
        h5file = h5py.File(root_path, "r")
        ignore_writers = {
            "f0048_00",
            "f0052_42",
            "f0741_44",
            "f0825_23",
            "f0848_42",
            "f1209_31",
            "f1432_44",
            "f1756_07",
            "f1767_34",
            "f1965_23",
            "f2027_41",
            "f2199_64",
        }
        writers = sorted(set(h5file.keys()) - ignore_writers)
        # load the data
        start = time.time()
        res_fed_x = []
        res_fed_y = []
        for i in range(args.num_clients):
            res_x, res_y = load_client(
                i, h5file, writers, cmin, args.ir, group, logger, args.os, channel, dim
            )
            res_fed_x.append(res_x)
            res_fed_y.append(res_y)
        logger.info(
            f"load {args.num_clients} clients dataset. ({time.time() - start:.2f})s"
        )
        # res_fed_x = []
        # res_fed_y = []
        # for cx, cy in zip(fed_x, fed_y):
        #     logger.info(f'{cx.shape}, {cy.shape}')
        #     res_x, res_y = resampling(cx, cy, sampling=args.os, len_lim=True, random=True)
        #     res_fed_x.append(res_x.reshape(-1, channel, dim, dim))
        #     res_fed_y.append(res_y.reshape(-1))
        # res_fed_x = np.concatenate(res_fed_x, axis=0)
        # res_fed_y = np.concatenate(res_fed_y, axis=0)
        # print(res_fed_x.shape, res_fed_y.shape)
        # convert to data loader
        fed_ds = [
            TensorDataset(torch.Tensor(res_fed_x[i]), torch.Tensor(res_fed_y[i]))
            for i in range(args.num_clients)
        ]
        # load the test data
        start = time.time()
        test_x, test_y = [], []
        for i in range(
            args.num_clients * group, min(len(writers), 2 * args.num_clients * group)
        ):
            h5group = h5py.Group(h5file[writers[i]].id)
            x, y = np_data(h5group, cmin)
            test_x.append(x)
            test_y.append(y)
        test_x = np.concatenate(test_x, axis=0).reshape(-1, channel, dim, dim)
        test_y = np.concatenate(test_y, axis=0)
        test_ds = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        logger.info(
            f"load the testset of {args.num_clients*group} writers. ({time.time() - start:.2f})s"
        )
        return fed_ds, test_ds
    elif args.dataset == "cifar10":
        cmin = np.random.randint(10)
        x_train, x_test, y_train, y_test = load_vision_data(args.dataset)
        # convert to numpy
        if isinstance(x_train, torch.Tensor):
            x_train, x_test = x_train.numpy(), x_test.numpy()
            y_train, y_test = y_train.numpy(), y_test.numpy()
        else:
            y_train, y_test = np.array(y_train), np.array(y_test)
        # get the original shape
        n, _ = x_train.shape[0], x_train.shape[1:]
        x_train = x_train.reshape(n, -1)
        x_test = x_test.reshape(-1, channel, dim, dim)
        y_test = y_test.reshape(-1)
        # select and set the minority class to 1
        y_train = (y_train == cmin).astype("int")
        y_test = (y_test == cmin).astype("int")
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        rid = np.random.permutation(y_train.shape[0])
        x_train, y_train = x_train[rid], y_train[rid]

        # split the data into clients
        n_min = int((y_train == 1).sum() / args.ir / args.num_clients)
        n_maj = int((y_train == 0).sum() / args.num_clients)
        min_idx = np.where(y_train == 1)[0]
        maj_idx = np.where(y_train == 0)[0]
        cid = [
            np.concatenate(
                [
                    min_idx[i * n_min : (i + 1) * n_min],
                    maj_idx[i * n_maj : (i + 1) * n_maj],
                ]
            )
            for i in range(args.num_clients)
        ]
        fed_x = [x_train[cid[i]] for i in range(args.num_clients)]
        fed_y = [y_train[cid[i]] for i in range(args.num_clients)]

        # sample the test data
        res_fed_x = []
        res_fed_y = []
        for cx, cy in zip(fed_x, fed_y):
            res_x, res_y = resampling(
                cx, cy, sampling=args.os, len_lim=True, random=True
            )
            res_fed_x.append(res_x.reshape(-1, channel, dim, dim))
            res_fed_y.append(res_y.reshape(-1))
        # res_fed_x = np.concatenate(res_fed_x, axis=0)
        # res_fed_y = np.concatenate(res_fed_y, axis=0)
        # print(res_fed_x.shape, res_fed_y.shape)
        # convert to data loader
        fed_ds = [
            TensorDataset(torch.Tensor(res_fed_x[i]), torch.Tensor(res_fed_y[i]))
            for i in range(args.num_clients)
        ]
        test_ds = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
        # raise NotImplementedError(f'Dataset: {args.dataset} not implemented yet')
    return fed_ds, test_ds
