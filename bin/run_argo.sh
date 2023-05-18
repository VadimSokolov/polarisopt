#!/bin/bash
#SBATCH --job-name=morris
#SBATCH --partition=all-HiPri
#SBATCH --nodes=1
#SBATCH --time=04:00:00
# NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/home/vsokolov/slurm/%x-%j.out  # Output file`
#SBATCH --error=/home/vsokolov/slurm/%x-%j.err   # Error file

module load  gcc/9.3.0
module load HDF/5/hdf5-1.12.0-gcc-8.4.0 
module load anaconda3/latest
source activate pol
export PYTHONPATH="$PYTHONPATH:/home/vsokolov/polaris-hpc"

python morris_SA.py  
