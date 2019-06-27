#!/bin/bash

timestamp=`date '+%Y%m%d_%H%M%S'`
source_directory=$1

time python bebop_preprocess.py $timestamp

time python bebop_submit.py $source_directory $timestamp

time python bebop_postprocess.py $timestamp

