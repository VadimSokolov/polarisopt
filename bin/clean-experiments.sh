#!/bin/bash

# for file in Sim*/Bloomington_iteration_1/Bloomington-Result.h5; do echo $file; mkdir -p /projects/vsokolov/$(dirname $file); cp $file /projects/vsokolov/$(dirname $file); done
# for file in Sim*/*.json; do echo $file; mkdir -p /projects/vsokolov/$(dirname $file); cp $file /projects/vsokolov/$(dirname $file); done
# for file in Sim*/Bloomington_iteration_1/log/*; do echo $file; mkdir -p /projects/vsokolov/$(dirname $file); cp $file /projects/vsokolov/$(dirname $file); done

# run_folder="Austin_iteration_10"
# res_file="Austin-"
# find . -maxdepth 1 ! -name '*.json' ! -name '*.slurm' ! -name '*.out' ! -name '*.err'  -type f -exec rm -f {} +
# find . -maxdepth 1 ! -name 'Austin-Demand.sqlite' ! -name 'Austin-Result.h5' ! -name 'summary.csv' -type f -exec rm -f {} +

cd /projects/vsokolov/austin-transit-focus/experiments
# d=Sim48/
for d in */ ; do
# for d in Sim0 Sim1 Sim2 ; do
    echo "$d"
    cd $d
    # find . -maxdepth 1 ! -name '*.json' ! -name '*.slurm' ! -name '*.out' ! -name '*.err'  -type f -exec rm -f {} +
    rm -rf  Austin_iteration_10
    rm -rf  log
    rm *.out
    rm *.err
    rm *.slurm
    # echo "$(pwd)" 
    # find . -maxdepth 1 ! -name 'Austin-Demand.sqlite' ! -name 'Austin-Result.h5' ! -name 'summary.csv' -type f -exec rm -f {} +
    cd /projects/vsokolov/austin-transit-focus/experiments
    # echo "$(pwd)" 
done
