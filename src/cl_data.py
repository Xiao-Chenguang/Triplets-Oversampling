import numpy as np
import pandas as pd
from torchvision import datasets
from .samplers import Triplets
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE


D_PATH = {'root': 'datasets/',
          'vehicle': 'vehicle.csv',
          'diabete': 'diabete.csv',
          'vowel': 'vowel.csv',
          'ionosphere': 'ionosphere.csv',
          'abalone': 'abalone.csv',
          'satimage': 'satimage.csv',
          'haberman': 'haberman.csv',
          'fraud': 'fraud.csv',
          'aloi': 'aloi.csv',
          'pulsar': 'pulsar.csv',
          }


D_PATH_VISION = {'root': 'datasets/',
                 'mnist': datasets.MNIST,
                 'cifar10': datasets.CIFAR10
                 }


SAMPLERS = {'oversampling': RandomOverSampler,
            'smote': SMOTE,
            'adasyn': ADASYN,
            'blsmote': BorderlineSMOTE
            }


CUSTOM_SAMPLERS = {'triplets': Triplets}


# load data file
def load_data(ds_name):
    '''
    ds_name: str, name of dataset
    '''
    data = pd.read_csv(
        D_PATH['root'] + D_PATH[ds_name], header=None).to_numpy(dtype='float32')
    return data[:, :-1], data[:, -1]


# load torchvision data
def load_vision_data(ds_name):
    '''
    ds_name: str, name of dataset
    '''
    train_ds = D_PATH_VISION[ds_name](D_PATH_VISION['root'], train=True)
    test_ds = D_PATH_VISION[ds_name](D_PATH_VISION['root'], train=False)
    x_train, y_train = train_ds.data, train_ds.targets
    x_test, y_test = test_ds.data, test_ds.targets
    return x_train, x_test, y_train, y_test


# train test split
def test_split(x, y, test_size=0.5, ir=1, normalize='minmax', seed=0):
    '''
    x: numpy array
    y: numpy array
    test_size: float, proportion of test set
    seed: int, random seed
    '''
    np.random.seed(seed)
    pidx = np.where(y == 1)[0]
    nidx = np.where(y == 0)[0]
    np.random.shuffle(pidx)
    np.random.shuffle(nidx)
    split = int(len(pidx) * (1 - test_size / ir)), int(len(nidx) * test_size)
    testid = np.concatenate((pidx[:split[0]], nidx[:split[1]]))
    trainid = np.concatenate((pidx[split[0]:], nidx[split[1]:]))

    x_test, x_train = x[testid], x[trainid]
    y_test, y_train = y[testid], y[trainid]
    # normalize the data into same scale
    if normalize == 'minmax':
        x_test = (x_test - x_train.min(0)) / (x_train.max(0) - x_train.min(0))  # normalize is vital for MLP
        x_train = (x_train - x_train.min(0)) / (x_train.max(0) - x_train.min(0))  # normalize is vital for MLP
    elif normalize == 'standard':
        x_test = (x_test - x_test.mean(0)) / x.std(0)  # standardize
        x_train = (x_train - x_train.mean(0)) / x.std(0)  # standardize
    else:
        raise ValueError(f'Unknown normalize method: {normalize}')
    return x_train, x_test, y_train, y_test


# do resampling early than convert to dataset.
def resampling(x, y, sampling='nonsampling', **kwargs):
    '''
    x: numpy array
    y: numpy array
    sampling: str, 'nonsampling', 'oversampling', 'smote', 'adasyn', 'blsmote', 'adsmote', 'triplets_bd'
    seed: int, random seed
    **kwargs: m_neighbors, n_neighbors, alpha
    '''

    if sampling == 'nonsampling':
        return x, y
    elif sampling in SAMPLERS:
        ros = SAMPLERS[sampling]()
        try:
            x_resampled, y_resampled = ros.fit_resample(x, y)
        except:
            x_resampled, y_resampled = x, y
    elif sampling in CUSTOM_SAMPLERS:
        sampler = CUSTOM_SAMPLERS[sampling](**kwargs)
        x_resampled, y_resampled = sampler.fit_resample(x, y)
    else:
        raise ValueError(f'Unknown sampling method: {sampling}.')
    # TODO: implement custom resampling methods

    return x_resampled, y_resampled
