#!/bin/bash

conda create -n urban-sim python=3.12 -y
conda activate urban-sim

pip3 install torch tensorboard tqdm hydra-core

apt install gdal-bin libgdal-dev # gdal-bin 不需要？
pip3 install GDAL # pip 安装没有识别本地 GDAL 版本，安装失败需要手动指定
