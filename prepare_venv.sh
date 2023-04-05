#!/bin/bash

# create a new conda environment
conda create -n trip python=3.9 -y

# activate the environment
conda activate trip

# install the requirements
echo "Installing the requirements"
pip install -r requirements.txt
