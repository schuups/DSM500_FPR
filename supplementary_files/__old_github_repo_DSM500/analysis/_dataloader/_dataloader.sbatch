#!/bin/bash

#SBATCH --job-name GC_Baseline
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --account a-csstaff

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500/github/analysis/_dataloader
    python _dataloader.py
"