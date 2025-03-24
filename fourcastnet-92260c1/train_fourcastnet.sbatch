#!/bin/bash

#SBATCH --job-name FCN_Baseline
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
##SBATCH --account a-csstaff
#SBATCH --time 01:00:00

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

srun --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/env_arm64.toml bash -c "
    source /iopsstor/scratch/cscs/stefschu/DSM500_FPR/env/venv_arm64/bin/activate
    cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/fourcastnet-92260c1

    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=./config/AFNO.yaml --config='afno_backbone' --run_num=0
"