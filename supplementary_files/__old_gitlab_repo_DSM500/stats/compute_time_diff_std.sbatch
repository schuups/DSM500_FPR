#!/bin/bash

#SBATCH --job-name td_recompute
#SBATCH --partition normal
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 4
#SBATCH --time 06:00:00

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/gitlab/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats
    python /iopsstor/scratch/cscs/stefschu/DSM500/github/examples/weather/graphcast/compute_time_diff_std.py
"
