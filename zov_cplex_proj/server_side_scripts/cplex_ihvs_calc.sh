#!/bin/bash

filename=$1
timestamp=$2
./cplex_master ./run_$timestamp/inputs/$filename  ./run_$timestamp/outputs

