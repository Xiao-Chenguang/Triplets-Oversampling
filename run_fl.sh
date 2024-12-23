#!/bin/bash

# change to the directory of this script
echo "Changing to the directory of this script"
cd $(dirname $0)

# create log directory if not exists
mkdir -p log
mkdir -p datasets/femnist
mkdir -p results

# unzip the femnist dataset if not exists
if [ ! -f ./datasets/femnist/write_digits.hdf5 ]; then
    echo "Unzipping the femnist dataset"
    gunzip ./write_digits.hdf5.gz -c > ./datasets/femnist/write_digits.hdf5
fi

# activate the environment
source .venv/bin/activate

# run the cnetralized learning
echo "Running the federated learning"
python run_fl_nn.py
