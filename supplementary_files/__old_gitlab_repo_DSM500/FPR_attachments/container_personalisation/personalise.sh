#!/bin/bash

# This script is used to personalise the container. 
# It is to be executed once within the container which mounts the host directories as per toml file.

##### Create venv
# Become root, if not already and install the necessary to create venvs
unshare -r
apt update && apt install -y python3-venv
exit # exit root
# Create the venv
cd /iopsstor/scratch/cscs/stefschu/DSM500/gitlab
python3 -m venv --system-site-packages venv_arm64
# Fix name overlap between the installed version of Modulus and the version in the container, as per https://setuptools.pypa.io/en/latest/userguide/development_mode.html#how-editable-installations-work
fix="mv /usr/local/lib/python3.10/dist-packages/modulus /usr/local/lib/python3.10/dist-packages/original_modulus 2>/dev/null"
echo $fix >> /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/venv_arm64/bin/activate
source /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/venv_arm64/bin/activate

##### Install the forked repository of Modulus.
##### Together with the 'mv' appended above, this overwrites the version persent in the container)
pip install -e /iopsstor/scratch/cscs/stefschu/DSM500/github
# At this point the following command should print 0.10.0a0, before it'd have been 0.9.0
# python -c "import modulus; print(modulus.__version__)"

##### Install dependencies required to run Modulus
# Despite not using MLFlow its library is imported in the code, so this module needs to be installed
pip install mlflow
# Install flashattention-v3, as suggested early in the training logs
pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper 
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
