import os

import requests

if not os.path.exists("datasets/raw/"):
    os.makedirs("datasets/raw")

urls = {
    "vehicle": "https://www.openml.org/data/download/54/dataset_54_vehicle.arff",
    "diabete": "https://www.openml.org/data/download/22044302/diabetes.arff",
    "vowel": "https://www.openml.org/data/download/52210/phpd8EoD9",
    "ionosphere": "https://www.openml.org/data/download/59/dataset_59_ionosphere.arff",
    "abalone": "https://www.openml.org/data/download/3620/dataset_187_abalone.arff",
    "satimage": "https://www.openml.org/data/download/3619/dataset_186_satimage.arff",
    "haberman": "https://www.openml.org/data/download/43/dataset_43_haberman.arff",
    "aloi": "https://www.openml.org/data/download/16787469/phpYaqmhm",
    "pulsar": "https://www.openml.org/data/download/22102202/dataset",
}

for dataset, url in urls.items():
    if os.path.exists(f"datasets/raw/{dataset}.arff"):
        print(f"[{dataset}] already downloaded")
        continue
    r = requests.get(url, allow_redirects=True)
    open(f"datasets/raw/{dataset}.arff", "wb").write(r.content)
    print(f">{dataset}< downloaded successfully")
