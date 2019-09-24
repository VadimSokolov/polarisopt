#!/bin/bash

set -e

./check_python.sh

timestamp=`date '+%Y%m%d_%H%M%S'`
base_directory=$1

source_directory=$base_directory/ZOV

run_directory=$base_directory/run_$timestamp
mkdir $run_directory

input_directory=$run_directory/inputs
mkdir $input_directory
results_directory=$run_directory/results
mkdir $results_directory

./concat_inputs.sh $source_directory $input_directory

time python bebop_preprocess.py $timestamp

time python bebop_submit.py $input_directory $timestamp

time python bebop_postprocess.py $results_directory $timestamp

./fix_files.sh $results_directory

