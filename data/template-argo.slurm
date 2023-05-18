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

module load  gcc/9.3.0
module load HDF/5/hdf5-1.12.0-gcc-8.4.0 
module load anaconda3/latest
source activate pol
$SCRIPT