#!/bin/bash

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

conda init bash
source ~/.bashrc

conda env create -f environment.yml

# Install sphinx
sudo apt-get update -y
sudo apt-get install python3-sphinx -y

pip install sphinx_rtd_theme
