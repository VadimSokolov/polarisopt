#!/bin/bash

function submit_cplex
{
	echo Calculating CPLEX on folder $1 with timestap=$2
	time python bebop_submit.py $1 $2
	rm  ~/polaris/cplex/input_divided/*	
}

function split_inputs
{
	source_directory=$1
	staging_directory=$2

	mkdir $staging_directory/input_0
	for f in $(find $source_directory -name '*_[0-9].txt')
	do
	    #echo $f
	    cp $f $staging_directory/input_0
	done
	for f in $(find $source_directory -name '*_[0-9][0-9].txt')
	do
	    #echo $f
	    cp $f $staging_directory/input_0
	done

	for i in {1..99..1}
	do
		num=$(find $source_directory -name "*_$i[0-9][0-9].txt"|wc -l)
		if ! [ $num = 0 ]; then
			mkdir $staging_directory/input_$i
			for f in $(find $source_directory -name *_$i[0-9][0-9].txt)
			do
				cp $f $staging_directory/input_$i
			done
		fi
	done
}

#run_cplex input

timestamp=`date '+%Y%m%d_%H%M%S'`
source_directory=$1
staging_directory=$2_$timestamp
mkdir $staging_directory
split_inputs $source_directory $staging_directory

time python bebop_preprocess.py $timestamp

#for d in $(find ~/polaris/cplex/input/ -mindepth 1 -maxdepth 1 -type d)
for d in $(find $staging_directory -mindepth 1 -maxdepth 1 -type d)
do
  #Do something, the directory is accessible with $d:
  submit_cplex $d $timestamp
done

time python bebop_postprocess.py $timestamp

