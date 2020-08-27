#!/bin/bash

set -e

./check_python.sh

input_directory=$1

time python run_cplex.py $input_directory

