#!/bin/bash

# change to the directory of this script
echo "Changing to the directory of this script"
cd $(dirname $0)

# activate the environment
conda activate trip

# run the cnetralized learning
echo "Running the federated learning"
python run_fl.py
