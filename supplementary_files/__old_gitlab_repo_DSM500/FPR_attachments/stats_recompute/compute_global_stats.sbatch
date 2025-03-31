#!/bin/bash

#SBATCH --job-name gtest
#SBATCH --partition normal
#SBATCH --nodes 39
#SBATCH --ntasks-per-node 21
#SBATCH --time 00:05:00

export HDF5_USE_FILE_LOCKING=FALSE

srun --unbuffered --mpi=pmix --environment=/iopsstor/scratch/cscs/stefschu/DSM500/gitlab/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats
    
    python compute_global_stats_with_sst_fix.py
    # python compute_global_stats_without_sst_fix.py
"
