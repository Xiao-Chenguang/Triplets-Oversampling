# Triplets-Oversampling-for-Federated-Datasets
Project for paper "Triplets Oversampling for Federated Datasets" submitted to ECML-PKDD2023

To reprocude the results of the paper, you need to follow the procedure below:
1. clone or copy the repository to your local machine
2. Open a terminal and go to the folder "Triplet_Oversampling"
3. run the scripts "run.sh" and "run_oversampling.sh" to generate the results of the paper.

## Quick start
You can reproduce the results with provided scripts.

Prepare the virtual environment and packages with conda:
*source prepare_venv.sh*

Run the centralized learning experiment:
*source ./run_cl.sh*

Run the federated learning experiment:
*source ./run_fl.sh*


Python environment and dependencies:
- Python 3.9.7
- Required package is listed in the file "requirements.txt"

It is encouraged to use a virtual environment to run the scripts.
For example, with conda:
*conda create -n trip python=3.9*

To activate the environment:
*conda activate trip*

To install the required packages:
*pip install -r requirements.txt*

And then you can run the scripts:
*./run_cl.sh*