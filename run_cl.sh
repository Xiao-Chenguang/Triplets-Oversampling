#!/bin/bash

# change to the directory of this script
echo "Changing to the directory of this script"
cd $(dirname $0)

# activate the environment
conda activate trip

# download the data
echo "Downloading the data"
python src/cl_download_data.py

# convert the data to csv file
echo "Converting the data to csv file"
python src/cl_to_binary_csv.py

# run the cnetralized learning
echo "Running the centralized learning"
python run_cl_svm.py
