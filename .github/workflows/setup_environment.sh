#!/bin/bash

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda init bash

conda env create -f environment.yml

# Install sphinx
sudo apt update -y
sudo apt install python3-sphinx
