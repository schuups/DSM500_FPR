#!/bin/bash

#SBATCH --job-name gtest
#SBATCH --partition normal
#SBATCH --account a-csstaff
#SBATCH --nodes 39
#SBATCH --ntasks-per-node 21
#SBATCH --time 00:05:00

export HDF5_USE_FILE_LOCKING=FALSE

srun --unbuffered --mpi=pmix --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation
    python compute_global_stats.py
"
