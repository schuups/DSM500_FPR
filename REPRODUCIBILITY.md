# Reproducibility notes

This file contains the (raw) notes on the step followed for project environment setup and execution. Although they might not appear as well organised, they should be sufficient in supoprting the reader interested in reproducing results. 

## Prepare Github repo
```sh
# Create new private github repo, and invite the GoldSmithsMScDataScience user

# On Alps, clone repo
cd /iopsstor/scratch/cscs/stefschu
git clone git@github.com:schuups/DSM500_FPR.git
cd DSM500_FPR
```

## Clone and include Modulus code setup

```sh
# Get Modulus repository content (a5275d8 represents HEAD of the main branch on Dec 26th 2024)
wget https://github.com/NVIDIA/modulus/archive/a5275d8.zip
unzip a5275d8.zip
mv modulus-a5275d81ea562a80cf16e059c204c8a59fc4ddd9 modulus-a5275d8
rm a5275d8.zip
rm -rf modulus-a5275d8/.git

$> du -sh modulus-a5275d8
98M     modulus-a5275d8
```

## Link training data
```sh
# Download via modulus as per instructions found under https://github.com/NVlabs/FourCastNet

cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0
ln -s $PATH_OF_SYSTEM_SHARED_DATASET_ON_CSCS_STORAGE/train train
ln -s $PATH_OF_SYSTEM_SHARED_DATASET_ON_CSCS_STORAGE/out_of_sample out_of_sample
ln -s $PATH_OF_SYSTEM_SHARED_DATASET_ON_CSCS_STORAGE/test test
```


## Development environment setup
```sh
mkdir env
cd env

# Prepare container image (each rank trying to pull from the internet each time will not work when running at-scale)
enroot import -o modulus-ngc-24.12-arm64.sqsh docker://nvcr.io#nvidia/modulus/modulus:24.12

# Define environment file for jobs execution
$> cat env_arm64.toml 
image = "/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/modulus-ngc-24.12-arm64.sqsh"

mounts = [
    "/iopsstor/scratch/cscs/stefschu:/iopsstor/scratch/cscs/stefschu",
    "/users/stefschu/.bash_history:/users/stefschu/.bash_history",
    "/users/stefschu/.netrc:/users/stefschu/.netrc",
    "/users/stefschu/.alias:/users/stefschu/.bash_aliases"
]

workdir = "/iopsstor/scratch/cscs/stefschu/DSM500_FPR"
writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

# Creation of the venv for container personalisations
srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml --job-name env_setup --pty bash

# Once in the container, execute the following steps to create a venv
## Become root, if not already and install the necessary to create venvs
unshare -r
apt update && apt install -y python3-venv
exit
##  Create the venv
cd  /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env
python3 -m venv --system-site-packages venv_arm64
## Fix name overlap between the installed version of Modulus and the version in the container, as per https://setuptools.pypa.io/en/latest/userguide/development_mode.html#how-editable-installations-work
fix="mv /usr/local/lib/python3.10/dist-packages/modulus /usr/local/lib/python3.10/dist-packages/original_modulus 2>/dev/null"
echo $fix >> /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate

## Install Modulus from downloaded files
## Together with the 'mv' appended above, this overwrites the version persent in the container)
# pip install -e /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-a5275d8
# The above is not desired anymore, as the experiments should be self contained (required files are copied manually from the modulus codebase)
fix="mv /usr/local/lib/python3.10/dist-packages/modulus /usr/local/lib/python3.10/dist-packages/original_modulus 2>/dev/null"
echo $fix >> /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate

# As a result, the following is expected:
(venv_arm64) root@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500_FPR# python
Python 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import modulus
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'modulus'
>>>

## At this point the following command should print 0.10.0a0, before it'd have been 0.9.0
## python -c "import modulus; print(modulus.__version__)"

## Install dependencies required to run Modulus
## Despite not using MLFlow its library is imported in the code, so this module needs to be installed
pip install mlflow
# Install for the stats computation
pip install mpi4py==3.1.6
# For interactive 3D plotting in Jupyter
pip install ipympl
# For plotting images on a sphere
pip install scikit-image
# For cartography information
pip install cartopy
# Others
pip install --upgrade ipywidgets
## Fix the following warnings, considering https://github.com/Dao-AILab/flash-attention/blob/7a802796e12ab36fc9830089c418497c1552bd53/hopper/setup.py#L36 I am fixing the commit to use (subsequent ones go OOM, and need further investigation)
## /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py:108: UserWarning: To use flash-attn v3, please use the following commands to install: 
## (1) pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper" 
## (2) python_path=`python -c "import site; print(site.getsitepackages()[0])"` 
## (3) mkdir -p $python_path/flashattn_hopper 
## (4) wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
pip install "git+https://github.com/Dao-AILab/flash-attention.git@36dddb891c152c381f85878198769aaa99261bec#egg=flashattn-hopper&subdirectory=hopper" 
python_path=`python -c "import site; print(site.getsitepackages()[0])"` 
mkdir -p $python_path/flashattn_hopper 
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/36dddb891c152c381f85878198769aaa99261bec/hopper/flash_attn_interface.py
# To build videos
pip uninstall opencv-python opencv-contrib-python -y
pip cache purge
pip install --no-cache-dir opencv-python opencv-contrib-python

## Exit container and release node allocation

## Check that things are as they should
srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml --job-name env_setup --pty bash
## Once in the container execute the following
source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
python -c "import modulus; print(modulus.__file__)"
## It should print the path to the cloned Modulus repo, instead of the version embedded in the container

# At this point the situation should be as follows
[clariden][stefschu@clariden-ln001 DSM500_FPR]$ ll -lsa
total 40
4 drwxrwxr-x+ 6 stefschu csstaff 4096 Mar 13 13:04 ./
4 drwxrwxr-x+ 8 stefschu csstaff 4096 Mar 13 12:50 ../
4 drwxrwxr-x+ 3 stefschu csstaff 4096 Mar 13 12:51 data/
4 drwxrwxr-x+ 3 stefschu csstaff 4096 Mar 13 13:04 env/
4 drwxrwxr-x+ 7 stefschu csstaff 4096 Mar 13 12:37 .git/
4 -rw-rw-r--+ 1 stefschu csstaff  153 Mar 13 12:55 .gitignore
4 drwxr-xr-x+ 8 stefschu csstaff 4096 Dec 20 21:54 modulus-a5275d8/
4 -rw-rw-r--+ 1 stefschu csstaff  254 Mar 13 12:45 README.md
8 -rw-rw-r--+ 1 stefschu csstaff 5791 Mar 13 13:12 REPRODUCIBILITY.md
[clariden][stefschu@clariden-ln001 DSM500_FPR]$
```

## Setup FourCastNet

```sh
git clone https://github.com/NVlabs/FourCastNet.git fourcastnet-92260c1
rm -rf fourcastnet-92260c1/.git
```