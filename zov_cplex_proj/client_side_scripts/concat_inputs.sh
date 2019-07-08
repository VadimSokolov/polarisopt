#!/bin/bash

input_dir=$1
output_dir=$2

#mkdir ~/polaris/cplex/input/input_0
for f in $(find $input_dir -name '*_[0-9].txt')
do
	#echo $f
	cat $f >> $output_dir/all_0.txt
done
for f in $(find $input_dir -name '*_[0-9][0-9].txt')
do
	#echo $f
	cat $f >> $output_dir/all_0.txt
done

for i in {1..20..1}
do
	#mkdir ~/polaris/cplex/input/input_$i
	#echo find file *_$i[0-9][0-9].txt
	for f in $(find $input_dir -name *_$i[0-9][0-9].txt)
	do
		#echo $f
		cat $f >> $output_dir/all_$i.txt
	done
done
