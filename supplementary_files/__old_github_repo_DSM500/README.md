# DSM500 notes

## Setup training data
```sh
cd /iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0

ln -s /iopsstor/scratch/cscs/stefschu/FCN/FCN_ERA5_data_v0/train train
ln -s /iopsstor/scratch/cscs/stefschu/FCN/FCN_ERA5_data_v0/out_of_sample out_of_sample
ln -s /iopsstor/scratch/cscs/stefschu/FCN/FCN_ERA5_data_v0/test test
```

## Development environment setup

The following are the steps executed to prepare the environment used from the dry run phase onwards. Nodes on `clariden` (a vCluster of Alps) are arm based.

```sh
# Create new private github repo, and invite the GoldSmithsMScDataScience user

# On Alps, clone repo
cd /iopsstor/scratch/cscs/stefschu/DSM500
git clone git@github.com:schuups/DSM500.git github
cd github

# Initial commit
echo "# DSM500
" > README.md
git config --local user.name "Schups"
git config --local user.email "schups@gmail.com"
git add .
git commit -m "Initial commit"
git push

# Branch
git checkout -b DryRun

# Get Modulus repository content (a5275d8 represents HEAD of the main branch on Dec 26th 2024)
wget https://github.com/NVIDIA/modulus/archive/a5275d8.zip
unzip a5275d8.zip
mv modulus-a5275d81ea562a80cf16e059c204c8a59fc4ddd9 modulus-a5275d8
rm a5275d8.zip

# For storage and project folder usability parts of modulus not relevant for this project are deleted
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> du -sh modulus-a5275d8
98M     modulus-a5275d8
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> du -sh modulus-a5275d8/*
12K     modulus-a5275d8/CHANGELOG.md
4.0K    modulus-a5275d8/CITATION.cff
12K     modulus-a5275d8/CONTRIBUTING.md
12K     modulus-a5275d8/Dockerfile
69M     modulus-a5275d8/docs
9.8M    modulus-a5275d8/examples
4.0K    modulus-a5275d8/FAQ.md
12K     modulus-a5275d8/LICENSE.txt
4.0K    modulus-a5275d8/Makefile
2.7M    modulus-a5275d8/modulus
4.0K    modulus-a5275d8/pyproject.toml
20K     modulus-a5275d8/README.md
16M     modulus-a5275d8/test
# After various deletions (incl. other models and other examples)
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> du -sh modulus-a5275d8/*
216K    modulus-a5275d8/examples
1.9M    modulus-a5275d8/modulus
4.0K    modulus-a5275d8/pyproject.toml

# Software environment definition file creation, i.e. "env_arm64.toml" and importing the relevant docker image
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> enroot import -o modulus-ngc-24.12-arm64.sqsh docker://nvcr.io#nvidia/modulus/modulus:24.12
...
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> ll -h 
total 18G
drwxrwxr-x+ 6 stefschu csstaff 4.0K Jan 11 16:29 ./
drwxr-x---+ 6 stefschu csstaff 4.0K Jan 11 15:37 ../
drwxrwxr-x+ 8 stefschu csstaff 4.0K Jan 11 16:28 .git/
-rw-rw-r--+ 1 stefschu csstaff   36 Jan 11 15:40 .gitignore
-rw-rw-r--+ 1 stefschu csstaff 6.7K Jan 11 16:32 README.md
-rw-rw-r--+ 1 stefschu csstaff  520 Jan 11 15:52 env_arm64.toml
drwxrwxr-x+ 3 stefschu csstaff 4.0K Jan 11 16:29 experiments/
drwxr-xr-x+ 5 stefschu csstaff 4.0K Jan 11 16:29 modulus-a5275d8/
-rw-r--r--+ 1 stefschu csstaff  18G Dec 31 00:10 modulus-ngc-24.12-arm64.sqsh
drwxrwxr-x+ 5 stefschu csstaff 4.0K Jan 11 15:53 venv_arm64/
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github> cat env_arm64.toml 
image = "/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-ngc-24.12-arm64.sqsh"
mounts = [
    "/iopsstor:/iopsstor",
    "/users/stefschu/.bash_history:/users/stefschu/.bash_history",
    "/users/stefschu/.netrc:/users/stefschu/.netrc",
    "/users/stefschu/.alias:/users/stefschu/.bash_aliases"
]
workdir = "/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast"
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github>

# Creation of the venv for container personalisations
srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml --job-name gtest --pty bash

# Once in the container, execute the following steps to create a venv
## Become root, if not already and install the necessary to create venvs
unshare -r
apt update && apt install -y python3-venv
exit
##  Create the venv
cd /iopsstor/scratch/cscs/stefschu/DSM500/github
python3 -m venv --system-site-packages venv_arm64
## Fix name overlap between the installed version of Modulus and the version in the container, as per https://setuptools.pypa.io/en/latest/userguide/development_mode.html#how-editable-installations-work
fix="mv /usr/local/lib/python3.10/dist-packages/modulus /usr/local/lib/python3.10/dist-packages/original_modulus 2>/dev/null"
echo $fix >> /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate

## Install Modulus from downloaded files
## Together with the 'mv' appended above, this overwrites the version persent in the container)
pip install -e /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8
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
srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml --job-name gtest --pty bash
## Once in the container execute the following
source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
python -c "import modulus; print(modulus.__version__)"
## It should print "0.10.0a0" and temporarily changing the version in modulus-a5275d8/modulus/__init__.py should be reflected
```

---

## Setup development container

```sh
# Build vtk with web support


# Create container under /dev/shm/stefschu/enrootdata
enroot create --name modulus-24.12-arm64 /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-ngc-24.12-arm64.sqsh

# Log into the container
ENROOT_SLURM_HOOK=off enroot start \
        --root \
        --rw \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --mount /iopsstor:/iopsstor \
        --mount /capstor:/capstor \
        --mount /users/stefschu/.bash_history:/root/.bash_history \
        --mount /users/stefschu/.netrc:/root/.netrc \
        --mount /users/stefschu/.alias:/root/.bash_aliases \
        --mount /users/stefschu/local:/root/local \
        --env SHM_SIZE=60g \
        --env MEMLOCK=-1 \
        --env STACK=67108864 \
        modulus-24.12-arm64 /bin/bash

# Jupyter usage
cd /iopsstor/scratch/cscs/stefschu/DSM500/github/analysis
source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
python -m ipykernel install --user --name=venv_arm64
jupyter lab --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.ip='0.0.0.0' --NotebookApp.port=8888 --NotebookApp.open_browser=False --NotebookApp.custom_display_url="http://$(hostname):8888/" --notebook-dir=$(pwd)

# Training via CLI (NB: wb_mode=disabled)
cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/02-Baseline
python /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast/train_graphcast.py --config-path $(pwd) wb_mode=disabled

# Simuluate multi-rank:
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast/train_graphcast_new_model.py --config-path $(pwd) wb_mode=disabled

# Once finished, especially on the login node
enroot remove modulus-24.12-arm64

```

---

## Training execution via CLI

```sh
mkdir -p /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun
cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun

# Copy default configuration
cp /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast/conf/config.yaml .

# Add sbatch script
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun$ cat dry_run.sbatch
#!/bin/bash

#SBATCH --job-name GC_DryRun
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --account a-csstaff

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source $BASE/venv_arm64/bin/activate
    cd $BASE/experiments/01-DryRun
    python $BASE/modulus-a5275d8/examples/weather/graphcast/train_graphcast.py --config-path $(pwd)
"
stefschu@clariden-ln001:/iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun>

srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml --job-name gtest -A a-csstaff --partition debug --pty bash
source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
# To run the training script in interactive mode, log into a container and mock the distribution manager settings by etting the following envs:
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 # default from torch launcher
# Then the following will not stall until a timeout error
/iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun$ python /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast/train_graphcast.py --config-path $(pwd) wb_mode=disabled
```

---

## 01-DryRun fixes

Notes of other steps required to fix things to make the dry run successful

```sh
# The metadata json file can be regenerated through the /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/dataset_metadata_recreation/99_curated_ERA5_field_names_reverse_engineering.ipynb notebook

# Place a "copy" of it with the dataset
cd /iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/dataset_metadata_recreation/metadata.json metadata.json

stefschu@clariden-ln003:/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0> ll
total 32
drwxr-x---+ 4 stefschu csstaff 4096 Jan 18 14:07 ./
drwxr-x---+ 3 stefschu csstaff 4096 Dec 30 13:56 ../
...
lrwxrwxrwx  1 stefschu csstaff  109 Jan 18 14:07 metadata.json -> /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/dataset_metadata_recreation/metadata.json
...


# The stats files can be regenerated through the scripts found under /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation
# The folder also contains the notebook used to prepare the scripts, which highlight the issue with the 21st channel.

# Once generated, the can be "copied" into the right places
# This linking is also necessary to swap the original version or the one with the fix (later on) 
cd /iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0/stats
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/global_means_no_sst_fix.npy global_means.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/global_stds_no_sst_fix.npy global_stds.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/time_diff_mean_new.npy time_diff_mean.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/time_diff_std_new.npy time_diff_std.npy

stefschu@clariden-ln003:/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0/stats> ll
total 2596
drwxrwxr-x+ 2 stefschu csstaff    4096 Jan 18 14:15 ./
drwxr-x---+ 4 stefschu csstaff    4096 Jan 18 14:07 ../
lrwxrwxrwx  1 stefschu csstaff     118 Jan 18 14:15 global_means.npy -> /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/global_means_no_sst_fix.npy
lrwxrwxrwx  1 stefschu csstaff     117 Jan 18 14:15 global_stds.npy -> /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/global_stds_no_sst_fix.npy
lrwxrwxrwx  1 stefschu csstaff     113 Jan 18 14:15 time_diff_mean.npy -> /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/time_diff_mean_new.npy
lrwxrwxrwx  1 stefschu csstaff     112 Jan 18 14:15 time_diff_std.npy -> /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation/time_diff_std_new.npy
stefschu@clariden-ln003:/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0/stats> 


# The land_sea_mask.nc and geopotential.nc files can be downloaded anew from CDS https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels}

stefschu@clariden-ln003:/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0> ll -h static
total 3.3M
drwxr-x---+ 2 stefschu csstaff 4.0K Jan 18 14:27 ./
drwxr-x---+ 4 stefschu csstaff 4.0K Jan 18 14:07 ../
-rw-r-----+ 1 stefschu csstaff 2.6M Dec  1 18:32 geopotential.nc
-rw-r-----+ 1 stefschu csstaff 763K Dec  1 18:32 land_sea_mask.nc
stefschu@clariden-ln003:/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0>

```

---

## 02-Baseline

```sh
mkdir -p /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/02-Baseline
cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/02-Baseline

cp /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/config.yaml .
cp /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/dry_run.sbatch baseline.sbatch

# Update both yaml and sbatch files, from DryRun to Baseline

# Run a profile by prepending the following, and enabling the corresponding option in the configurations
srun ... bash -c "
nsys profile --trace=cuda,osrt,nvtx --output graphcast_profile_rank_\$SLURM_LOCALID python $BASE/modulus-a5275d8/examples/weather/graphcast/train_graphcast.py --config-path $(pwd) wb_mode=disabled
"
```

---

## Baseline model training

```sh

tmux new-session -d -s watchdog 'cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/02-Baseline && python3.11 watchdog.py'

```