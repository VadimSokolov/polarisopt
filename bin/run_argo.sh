#!/bin/bash
#SBATCH --job-name=morris
#SBATCH --partition=all-HiPri
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out  # Output file
#SBATCH --error=%x-%j.err   # Error file
#SBATCh --cpus-per-task=8

# module restore polaris
module load anaconda3/latest
source activate pol
export PYTHONPATH="$PYTHONPATH:."

python bin/morris_SA.py  
