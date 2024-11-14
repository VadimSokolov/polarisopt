#!/bin/bash
#SBATCH --job-name=KPI
#SBATCH --partition=all-LoPri
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --output=%x-%j.out  # Output file
#SBATCH --error=%x-%j.err   # Error file
#SBATCh --cpus-per-task=8

# module restore polaris
module load anaconda3/latest
source activate pol
export PYTHONPATH=/home/vsokolov/polarislib

python -u bin/kpi.py  
