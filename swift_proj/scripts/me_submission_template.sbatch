#!/bin/bash

#SBATCH --output=${EXP_DIR}/me_output.txt
#SBATCH --error=${EXP_DIR}/me_output.txt
#SBATCH --partition=${QUEUE}
#SBATCH --account=${PROJECT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${PPN}
#SBATCH -D ${EXP_DIR}

srun -n ${PROCS} ${CMD}