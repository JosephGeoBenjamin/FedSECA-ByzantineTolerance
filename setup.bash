#!/bin/bash -x

conda create -n "newfed" python=3.10.13 -y
conda activate newfed
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

conda install -y -c anaconda ipykernel
ipython kernel install --name "newfed" --user


### For requirements generation followed solution from https://stackoverflow.com/q/31684375
# pip3 install pipreqs
# pipreqs --savepath=requirements.txt
