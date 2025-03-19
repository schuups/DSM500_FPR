#!/bin/bash

#SBATCH --job-name gtest
#SBATCH --partition normal
#SBATCH --account a-csstaff
#SBATCH --nodes 39
#SBATCH --ntasks-per-node 21
#SBATCH --time 00:05:00

export HDF5_USE_FILE_LOCKING=FALSE

srun --unbuffered --mpi=pmix --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/00_stat_files_creation
    python compute_global_stats.py
"
