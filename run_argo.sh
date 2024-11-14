#!/bin/bash
#SBATCH --job-name=quantile-opt
#SBATCH --partition=all-LoPri
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --output=quantile-opt/%x-%j.out  # Output file
#SBATCH --error=quantile-opt/%x-%j.err   # Error file
#SBATCh --cpus-per-task=8

# module restore polaris
module load anaconda3/latest
source activate pol
export PYTHONPATH="$PYTHONPATH:."

python -u quantile-opt/manual_search.py  
