#!/bin/bash

#SBATCH --job-name GC_Baseline
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --account a-csstaff

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline

    # python train_graphcast.py --config-path $(pwd) --config-name "baseline"
"