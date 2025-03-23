#!/bin/bash

#SBATCH --job-name GC_SOp
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=1-100%6
#SBATCH --account a-csstaff
#SBATCH --time 00:30:00

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    export WANDB_MODE=offline
    export OMP_NUM_THREADS=1
    export http_proxy=http://proxy.cscs.ch:8080
    export https_proxy=http://proxy.cscs.ch:8080

    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline
    
    wandb agent --count 1 schups/DSM500_FPR/mpgn79e1
"