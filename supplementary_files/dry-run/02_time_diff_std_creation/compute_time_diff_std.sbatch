#!/bin/bash

#SBATCH --job-name td_recompute
#SBATCH --partition normal
#SBATCH --account a-csstaff
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 4
#SBATCH --time 06:00:00

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    sleep 10
    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/dry-run/02_time_diff_std_creation
    sleep 10
    python compute_time_diff_std.py --config-path /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-dry-run/conf
"
