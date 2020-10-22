#!/bin/bash

#SBATCH --job-name=polaris_conv
#SBATCH --account=polaris
#SBATCH --partition=knlall
#SBATCH --nodes=1

# Run My Program
control_file=$1
srun run_convergence.py $control_file
