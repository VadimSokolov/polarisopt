#!/usr/bin/python
# Filename: bebop_submit.py

import json
import socket
import sys
import os
from os import walk
import shutil
import bebop_parsl

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

	bebop_parsl.submit_cplex_jobs(input_dir, bebop_results, timestamp)


