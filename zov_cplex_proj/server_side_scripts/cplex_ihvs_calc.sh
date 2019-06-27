#!/bin/bash

input_filename=$1
output_filename=$2
timestamp=$3

mpirun -np 256 ./cplex_ihvs ./run_$timestamp/inputs/$input_filename  ./run_$timestamp/outputs/$output_filename

