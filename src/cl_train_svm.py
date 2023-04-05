import os
import yaml
import numpy as np
from sklearn.svm import SVC
from .cl_data import load_data, load_vision_data, resampling, test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score


default_conf = 'src/config.yaml'

def svm_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return acc, recall, precision, f1, gmean, auc, ap


def run(dsname, sampling, seed, device='cpu', test_size=0.5, ir=1, outexten='', m_neighbors=10, n_neighbors=5, **kwargs):
    with open(default_conf, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load data
    if dsname in {'mnist', 'cifar10', 'cifar100', 'fashionmnist', 'svhn'}:
        x_train, x_test, y_train, y_test = load_vision_data(dsname)
        x_train = x_train.float().flatten(start_dim=1) / 255.
        x_test = x_test.float().flatten(start_dim=1) / 255.
        y_train = (y_train == 0).long()
        y_test = (y_test == 0).long()
    else:
        x, y = load_data(dsname)
        x_train, x_test, y_train, y_test = test_split(
            x, y, test_size=test_size, ir=ir, seed=seed)
    x_train, y_train = resampling(
        x_train, y_train, sampling=sampling, m_neighbors=m_neighbors, n_neighbors=n_neighbors, **kwargs)

    # define the model

    model = SVC(kernel='rbf', **config['svm_params'][dsname])

    # create log file
    if not os.path.exists('results'): os.mkdir('results')
    fname = f"results/svm_results{'' if outexten == '' else '_' + outexten}.csv"
    str_keys, str_values = '', ''
    for k, v in kwargs.items():
        str_keys += f",{k}"
        str_values += f",{v}"
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write(
               f"dataset,ir,sampling,m_neighbors,n_neighbors{str_keys},seed,acc,recall,precision,f1,gmean,auc,ap\n")
    log_header = f"{dsname},{ir},{sampling},{m_neighbors},{n_neighbors}{str_values},{seed}"

    print(f"Start training {dsname} with {sampling} methods")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = svm_metric(y_test, y_pred)
    print("acc: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}, gmean: {:.3f}, auc: {:.3f}, ap: {:.3f}".format(*score))
    with open(fname, 'a') as f:
        f.write(f"{log_header},{','.join([str(s) for s in score])}\n")
 