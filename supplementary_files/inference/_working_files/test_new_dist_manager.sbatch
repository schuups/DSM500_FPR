#!/bin/bash

#SBATCH --job-name GC_Baseline
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 5
#SBATCH --account a-csstaff
#SBATCH --time 00:30:00

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/inference/_working_files

    python test_new_dist_manager.py
"