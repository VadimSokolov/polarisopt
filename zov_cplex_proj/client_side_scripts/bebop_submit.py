#!/usr/bin/python
# Filename: bebop_submit.py

import json
import socket
import sys
import os
from os import walk
import shutil
import bebop_parsl

def split_file(filename, output_dir):
	buffer = []
	buffer_num = -1
	with open(filename, 'r') as inputfile:
		for line in inputfile:
			if line.startswith("P,"):
				buffer_num += 1
				buffer.append("")
			#print('line belongs in buffer {} - {}'.format(buffer_num, line))
			buffer[buffer_num] += line
	#print(len(buffer))
	size = calculate_size(len(buffer), 60)
	#print(size)
	basename = os.path.basename(filename)
	#print(basename)
	base = os.path.splitext(basename)
	#print(base[0])
	#new_output_dir = output_dir+'/'+base[0]
	#print(new_output_dir)
	#if os.path.exists(new_output_dir):
	#	shutil.rmtree(new_output_dir)
	#os.makedirs(new_output_dir)
	file_num = 0;
	output_filename = output_dir + '/' + base[0] + '_' + str(file_num) + '.txt'
	of = open(output_filename, "w")
	buffer_num = 0
	for stuff in buffer:
		if buffer_num == size:
			of.close()
			file_num += 1
			buffer_num = 0
			output_filename = output_dir + '/' + base[0] + '_' + str(file_num) + '.txt'
			of = open(output_filename, "w")
		of.write(stuff)
		buffer_num += 1
	of.close();

def calculate_size(num, optimum):
	if num < optimum:
		return num

	num_jobs = num // optimum
	rem = num % optimum
	extra = 0;
	if not num_jobs == 0:
		extra = rem//num_jobs
	#print('Total={} optimum={} num Jobs={} rem={} extra={}'.format(num, optimum, num_jobs, rem, extra))
	return optimum+extra+1


if __name__ == "__main__":
	input_dir = sys.argv[1]
	timestamp = sys.argv[2]

	for fname in os.listdir(input_dir):
		if fname.endswith('.txt'):
			# do stuff on the file
			break
	else:
		print('Input files do not exist in {}'.format(input_dir))
		sys.exit(1)

	control_fname = "polaris.json"
	with open(control_fname) as control_file:
		json_data = json.load(control_file)

	bebop_input = json_data["input_directory"]
	bebop_results = json_data["results_directory"]

	for dir_path, dirnames, filenames in walk(input_dir):
		for file in filenames:
			local_input_file = input_dir + '/' + file
			split_file(local_input_file, bebop_input)

	bebop_parsl.submit_cplex_jobs(bebop_input, bebop_results, timestamp)


