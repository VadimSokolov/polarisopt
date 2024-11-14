#!/bin/bash
#SBATCH --job-name=latin
#SBATCH --partition=all-LoPri
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --output=latin-study/%x-%j.out  # Output file
#SBATCH --error=latin-study/%x-%j.err   # Error file
#SBATCh --cpus-per-task=8

# module restore polaris
module load anaconda3/latest
source activate pol
export PYTHONPATH="$PYTHONPATH:."

python -u latin-study/latin.py  
