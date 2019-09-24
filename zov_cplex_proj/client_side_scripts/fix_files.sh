#!/bin/bash

FILES=$1/*.txt
for file in $FILES
do
  echo $file
  sed  '/^$/d' $file > $1/temp_file_1
  sed  ':a;N;$!ba;s/\$\n/\$/g' $1/temp_file_1 > $1/temp_file_2
  rm $1/temp_file_1
  rm $file
  mv $1/temp_file_2 $file
done

rm $1/all_results.txt
