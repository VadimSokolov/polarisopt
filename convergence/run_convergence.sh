#!/bin/bash

#run_convergence.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <data_directory>"
  exit 1
fi

python3 run_convergence.py convergence_control_linux.json $1
