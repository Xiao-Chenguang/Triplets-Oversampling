# Triplets-Oversampling-for-Federated-Datasets
Project for paper "Triplets Oversampling for Federated Datasets" submitted to ECML-PKDD2023

To reprocude the results of the paper, you need to follow the procedure below:
1. Clone or copy the repository to your local machine
2. Open a terminal and go to the folder "Triplets_Oversampling_for_Federated_Datasets"
3. Follow [Quick start](#quick-start) to generate the results of the paper simply.

## Quick start
You can reproduce the results with provided scripts.

Prepare the virtual environment and packages with conda:
```bash
source ./prepare_venv.sh
```


Run the **centralized learning** experiment:
```bash
source ./run_cl.sh
```

Run the **federated learning** experiment:
```bash
source ./run_fl.sh
```


## Python environment and dependencies:
- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.9
- Required package is listed in the file "[requirements.txt](./requirements.txt)"

## Synthesis quality comparison
The source code for the synthesis quality comparison in the paper is also published.
You run the [jupyter notebook](./synthesis_quality.ipynb) to get the comporation figures yourself.

You may also customize the dataset you prefer to check the synthesis quality of selected sampling algorithms.