#!/bin/bash
#SBATCH --account=POLARIS
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --time=24:00:00
##SBATCH --mail-user=vsokolov@gmu.edu
##SBATCH --mail-type=FAIL
#SBATCH --wait
#NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=$OUTPUTFOLDER/%x-%j.out  # Output file`
#SBATCH --error=$OUTPUTFOLDER/%x-%j.err   # Error file
#SBATCh --cpus-per-task=$NCPUS
#SBATCH --mem=$MEM
#SBATCH --job-name=$JOBNAME


module load anaconda3/latest
$SCRIPT