#!/bin/bash

conda env create -f environment.yml
conda install -y -c anaconda ipykernel
ipython kernel install --name "jfed" --user
