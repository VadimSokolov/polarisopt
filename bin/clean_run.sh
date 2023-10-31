#!/bin/bash
# rm timedep_training_data.json 2> /dev/null
# rm -rf experiments
rm *.err 2> /dev/null
rm *.out 2> /dev/null
sbatch bin/run_argo.sh 
