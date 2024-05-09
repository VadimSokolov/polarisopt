#!/bin/bash
#SBATCH --partition=all-LoPri
#SBATCH --time=2-00:00:00
#SBATCH --wait
#NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=$OUTPUTFOLDER/%x-%j.out  # Output file`
#SBATCH --error=$OUTPUTFOLDER/%x-%j.err   # Error file
#SBATCh --cpus-per-task=$NCPUS
#SBATCH --mem=$MEM
#SBATCH --job-name=$JOBNAME

export PYTHONPATH=/home/vsokolov/polarislib
module load anaconda3/latest HDF/5/1.12.0
source activate pol
$SCRIPT