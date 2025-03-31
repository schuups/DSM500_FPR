#!/bin/bash

#SBATCH --job-name GC_HPO
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output=swipe_logs/%A_%a.out
#SBATCH --array=1-1000%50
##SBATCH --account a-csstaff
#SBATCH --time 00:10:00

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    export HDF5_USE_FILE_LOCKING=FALSE
    export HDF5_CHUNK_CACHE_SIZE=104857600
    export HDF5_SWMR_WRITE=1
    export HDF5_SWMR_READ=1
    export OMP_NUM_THREADS=1

    source $BASE/venv_arm64/bin/activate
    cd $BASE/experiments/02-Baseline
    export WANDB_MODE=offline
    wandb agent --count 1 schups/GraphCast/tl7e22l8
"