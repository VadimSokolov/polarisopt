#!/bin/bash

filename=$1
timestamp=$2
script_directory="$PWD"
outputs_directory="$PWD/run_$timestamp/outputs"
results_directory="$PWD/run_$timestamp/results"

cd $outputs_directory
find "$PWD" -name "*" -type f > $results_directory/all_results.txt
