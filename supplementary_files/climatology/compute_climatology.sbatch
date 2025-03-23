#!/bin/bash

#SBATCH --job-name GC_Climatology
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --time 06:00:00
##SBATCH --account a-csstaff

srun --ntasks=1 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    ource /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/climatology
    python compute_climatology.py --config-path /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline/conf --config-name config
"
