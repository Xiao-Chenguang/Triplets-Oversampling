#!/bin/bash

# change to the directory of this script
echo "Changing to the directory of this script"
cd $(dirname $0)

# activate the environment
source .venv/bin/activate

# run the cnetralized learning
echo "Running the federated learning"
python run_fl_nn.py
