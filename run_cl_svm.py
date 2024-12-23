from itertools import product

from src.cl_train_svm import run

# ================== all default ==================
datasets = [
    "vehicle",
    "diabete",
    "vowel",
    "ionosphere",
    "abalone",
    "satimage",
    "haberman",
    "aloi",
    "pulsar",
]
irs = [1, 2, 4, 8]
samplings = ["nonsampling", "oversampling", "adasyn", "smote", "blsmote", "triplets"]
seeds = range(30)

for params in product(datasets, irs, samplings, seeds):
    run(dsname=params[0], ir=params[1], sampling=params[2], seed=params[-1])
