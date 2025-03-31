#!/bin/bash

#SBATCH --job-name GC_Lab
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --time 23:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --account a-csstaff

srun bash -c "sleep infinity"