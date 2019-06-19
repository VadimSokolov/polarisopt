#!/bin/bash

filename=$1
timestamp=$2
script_directory="$PWD"
outputs_directory="$PWD/run_$timestamp/outputs"
results_directory="$PWD/run_$timestamp/results"
cd $outputs_directory
find "$PWD" -name "Activities_Trips_*" > $results_directory/all_files.txt

cd $results_directory
split -l 10000 --numeric-suffixes --additional-suffix=.txt  $results_directory/all_files.txt files

find "$PWD" -name "files*" > $results_directory/file_of_files.txt

cd $script_directory
./cplex_merge_master $results_directory/file_of_files.txt $results_directory

cd $results_directory
find . -type f -name "results_files*" -printf "%f\n" > all_results.txt
