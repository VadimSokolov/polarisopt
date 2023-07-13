#!/bin/bash
#SBATCH --partition=all-HiPri
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --wait
#NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=$OUTPUTFOLDER/%x-%j.out  # Output file`
#SBATCH --error=$OUTPUTFOLDER/%x-%j.err   # Error file
#SBATCh --cpus-per-task=$NCPUS
#SBATCH --mem=$MEM
#SBATCH --job-name=$JOBNAME

# module restore polaris
module load anaconda3/latest
source activate pol
$SCRIPT