#!/bin/bash

#SBATCH --job-name GC_Climatology
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --time 06:00:00
#SBATCH --account a-csstaff

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github


srun --ntasks=1 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source $BASE/venv_arm64/bin/activate
    cd $BASE/analysis/_climatology
    python compute_climatology.py --config-path /iopsstor/scratch/cscs/stefschu/DSM500/github/analysis --config-name config_new
"
