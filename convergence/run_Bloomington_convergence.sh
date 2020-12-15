#!/bin/bash

#SBATCH --job-name=polaris_conv
#SBATCH --account=polaris
#SBATCH --partition=knlall
#SBATCH --nodes=1
#SBATCH --time=03:00:00

module load anaconda3/2020.07

echo "Starting convergence..."
echo "See  /lcrc/project/POLARIS/bebop/polaris/data/Bloomington/convergence_results/simulation_out.log for output"
python3 run_convergence.py bloomington_bebop_control.json /lcrc/project/POLARIS/bebop/polaris/data/Bloomington
echo "Convergence completed."
