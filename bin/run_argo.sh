#!/bin/bash
#SBATCH --job-name=morris
#SBATCH --partition=all-HiPri
#SBATCH --nodes=1
#SBATCH --time=04:00:00
# NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/home/vsokolov/slurm/%x-%j.out  # Output file`
#SBATCH --error=/home/vsokolov/slurm/%x-%j.err   # Error file

# module restore polaris
module load anaconda3/latest
source activate pol
export PYTHONPATH="$PYTHONPATH:/home/vsokolov/pol/polaris-hpc"

cd /home/vsokolov/pol/polaris-hpc
# rm /gpfs/fs1/home/vsokolov/polaris-hpc/DRGP/data/timedep_training_data.json
python bin/morris_SA.py  
