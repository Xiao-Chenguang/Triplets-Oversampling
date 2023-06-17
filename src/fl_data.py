import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
from torchvision import datasets, transforms
from src.cl_data import load_vision_data, resampling


class ImbDataset(Dataset):
    def __init__(self, dataset, cmin, ir):
        self.dataset = dataset
        # get the index of dataset with target cmin
        min_id = np.where(np.array(dataset.targets) == cmin)[0]
        min_id = np.random.choice(min_id, int(len(min_id) / ir), replace=False)
        maj_id = np.where(np.array(dataset.targets) != cmin)[0]
        # set self.id equal the concatenation of min_id and maj_id
        self.id = np.concatenate((min_id, maj_id))

    def __getitem__(self, index):
        return self.dataset[self.id[index]]

    def __len__(self):
        return len(self.id)


def gray_img_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def get_fed_dataset(args, channel, dim):
    cmin = np.random.randint(10)
    if args.dataset == 'femnist':
        group = 20
        root_path = 'datasets/femnist/write_digits/'
        ignore_writers = {'f0048_00', 'f0052_42', 'f0741_44', 'f0825_23',
                          'f0848_42', 'f1209_31', 'f1432_44', 'f1756_07',
                          'f1767_34', 'f1965_23', 'f2027_41', 'f2199_64'}
        writers = sorted(set(os.listdir(root_path)) - ignore_writers)
        # load the data
        fed_ds = []
        for i in range(args.num_clients):
            client_ds = []
            client_tagets = []
            for j in range(i*group, (i+1) * group):
                tem_ds = datasets.ImageFolder(os.path.join(
                    root_path, writers[j]),
                    transform=transforms.ToTensor(),
                    loader=gray_img_loader)
                client_ds.append(tem_ds)
                client_tagets.extend(tem_ds.targets)
            grouped_ds = ConcatDataset(client_ds)
            grouped_ds.targets = client_tagets  # type: ignore
            fed_ds.append(ImbDataset(grouped_ds, cmin, args.ir))
        # load the test data
        test_ds = []
        for i in range(args.num_clients*group, len(writers)):
            tem_ds = datasets.ImageFolder(os.path.join(
                root_path, writers[i]),
                transform=transforms.ToTensor(),
                loader=gray_img_loader)
            test_ds.append(tem_ds)
        test_ds = ConcatDataset(test_ds)
        return fed_ds, test_ds
    elif args.dataset == 'cifar10':
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
        y_test = y_test.reshape(-1)
        # select and set the minority class to 1
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
            res_x, res_y = x_train, y_train = resampling(
                cx, cy, sampling=args.os, len_lim=True, random=True)
            res_fed_x.append(res_x.reshape(-1, channel, dim, dim))
            res_fed_y.append(res_y.reshape(-1))
        # res_fed_x = np.concatenate(res_fed_x, axis=0)
        # res_fed_y = np.concatenate(res_fed_y, axis=0)
        # print(res_fed_x.shape, res_fed_y.shape)
        # convert to data loader
        fed_ds = [TensorDataset(torch.Tensor(res_fed_x[i]), torch.Tensor(
            res_fed_y[i])) for i in range(args.num_clients)]
        test_ds = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    else:
        raise NotImplementedError('Dataset not implemented yet')
        # raise NotImplementedError(f'Dataset: {args.dataset} not implemented yet')
    return fed_ds, test_ds
