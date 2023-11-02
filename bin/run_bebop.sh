#!/bin/bash
#SBATCH --job-name=morris
#SBATCH --account=POLARIS
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --mail-user=vsokolov@gmu.edu
#SBATCH --mail-type=FAIL
# NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/home/vsokolov/slurm/%x-%j.out  # Output file`
#SBATCH --error=/home/vsokolov/slurm/%x-%j.err   # Error file

module restore polaris
module load anaconda3/2021.05 
source activate pol
export PYTHONPATH="$PYTHONPATH:/home/vsokolov/polaris-hpc"

cd /home/vsokolov/polaris-hpc
# rm /gpfs/fs1/home/vsokolov/polaris-hpc/DRGP/data/timedep_training_data.json
python -u bin/morris_SA.py  
