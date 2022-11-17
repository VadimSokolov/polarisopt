#!/bin/bash
#SBATCH --job-name=saem
#SBATCH --account=POLARIS
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=vsokolov@gmu.edu
#SBATCH --mail-type=FAIL
# NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/home/vsokolov/slurm/%x-%j.out  # Output file`
#SBATCH --error=/home/vsokolov/slurm/%x-%j.err   # Error file

module restore polaris
module load anaconda3/2021.05 
source activate pol

cd /home/vsokolov/polaris-hpc/DRGP
python morris_SA.py  
