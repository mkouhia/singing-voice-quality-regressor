#!/usr/bin/env bash

# Script to setup working environment on DataCrunch.io, that is
# running a pre-setup FastAI image on CUDA 11.3.

# Install system ffmpeg and sox
sudo apt install -y ffmpeg sox libsox-fmt-all

# Install Mamba for faster installs
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b

eval "$(conda shell.bash hook)"
conda activate fastai

# Fix ffmpeg installation
/home/user/mambaforge/bin/mamba update -y ffmpeg -c conda-forge

# Update torch packages to latest supporting cudatoolkit=11.3
/home/user/mambaforge/bin/mamba install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# Update fastai to latest
/home/user/mambaforge/bin/mamba update -y fastai -c fastchan

# Add project dependencies
/home/user/mambaforge/bin/mamba install -y dvc dvc-s3 youtube-dl pyarrow librosa

pip install colorednoise
pip install -U git+https://github.com/mkouhia/fastaudio.git@master --no-deps


echo "Setup done! please perform following:"
echo "vi .dvc/config.local  # add configuration for data storage"
echo "dvc pull"
