#!/bin/bash

# change to the directory of this script
cd $(dirname $0)

# download the data
python src/cl_download_data.py

# convert the data to csv file
python src/cl_convert_data.py

# run the cnetralized learning
python run_svm.py
