#!/usr/bin/env bash
set -e

# System deps
sudo apt update
sudo apt install -y \
  build-essential \
  libgl1-mesa-dev \
  libosmesa6-dev \
  mesa-utils \
  ffmpeg \
  patchelf

# Miniconda (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Create env
conda env create -f bpref39.yml
conda activate bpref39

# Rendering
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
