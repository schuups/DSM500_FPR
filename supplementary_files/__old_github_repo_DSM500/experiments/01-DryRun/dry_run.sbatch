#!/bin/bash

#SBATCH --job-name GC_DryRun
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --account a-csstaff

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source $BASE/venv_arm64/bin/activate
    cd $BASE/experiments/01-DryRun
    python $BASE/modulus-a5275d8/examples/weather/graphcast/train_graphcast.py --config-path $(pwd)
"