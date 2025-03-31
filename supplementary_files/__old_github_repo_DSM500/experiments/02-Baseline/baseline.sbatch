#!/bin/bash

#SBATCH --job-name GC_Baseline
#SBATCH --partition debug
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --account a-csstaff

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    export HDF5_USE_FILE_LOCKING=FALSE
    export HDF5_CHUNK_CACHE_SIZE=104857600
    export HDF5_SWMR_WRITE=1
    export HDF5_SWMR_READ=1

    source $BASE/venv_arm64/bin/activate
    cd $BASE/experiments/02-Baseline
    python train_graphcast.py --config-path . --config-name config
"