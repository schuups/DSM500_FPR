#!/bin/bash

#SBATCH --job-name GC_Inference
#SBATCH --partition debug
#SBATCH --nodes 2
#SBATCH --account a-csstaff
#SBATCH --exclude nid[005009,005013]

BASE=/iopsstor/scratch/cscs/stefschu/DSM500/github

####################################################################################################
# Stage 1: Generates the inference data and video frames plan
srun --ntasks=1 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD=SLURM
    source $BASE/venv_arm64/bin/activate
    cd $BASE/inference
    python stage1.py
"
previous_exit=$?
if [ $previous_exit -ne 0 ]; then
    echo "Error in stage1.py, exiting"
    exit $previous_exit
else
    echo "Stage 1 completed successfully"
fi

####################################################################################################
# Stage 2: Generate images of the globe (one for each combination of sample, channel, step and type: input, output or prediction)
srun --ntasks=200 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    export HDF5_USE_FILE_LOCKING=FALSE
    source $BASE/venv_arm64/bin/activate
    cd $BASE/inference
    python stage2.py
"
previous_exit=$?
if [ $previous_exit -ne 0 ]; then
    echo "Error in stage2.py, exiting"
    exit $previous_exit
else
    echo "Stage 2 completed successfully"
fi

####################################################################################################
# Stage 3: Generate video frames as images
srun --ntasks=200 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    export HDF5_USE_FILE_LOCKING=FALSE
    source $BASE/venv_arm64/bin/activate
    cd $BASE/inference
    python stage3.py
"
previous_exit=$?
if [ $previous_exit -ne 0 ]; then
    echo "Error in stage3.py, exiting"
    exit $previous_exit
else
    echo "Stage 3 completed successfully"
fi

####################################################################################################
# Stage 4: Generate video and cleaup
srun --ntasks=1 --unbuffered --environment=/iopsstor/scratch/cscs/stefschu/DSM500/github/env_arm64.toml bash -c "
    source $BASE/venv_arm64/bin/activate
    cd $BASE/inference
    python stage4.py
"
previous_exit=$?
if [ $previous_exit -ne 0 ]; then
    echo "Error in stage4.py, exiting"
    exit $previous_exit
else
    echo "Stage 4 completed successfully"
fi