#!/bin/bash

#SBATCH --job-name td_recompute
#SBATCH --partition normal
#SBATCH --account a-csstaff
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 4
#SBATCH --time 06:00:00

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    sleep 10
    source /iopsstor/scratch/cscs/stefschu/DSM500/github/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/01-DryRun/stats_files_recreation
    sleep 10
    python /iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast/compute_time_diff_std.py --config-path $(pwd)/..
"
