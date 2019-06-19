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
	timestamp = sys.argv[1]

	control_fname = "polaris.json"
	with open(control_fname) as control_file:
		json_data = json.load(control_file)

	bebop_input = json_data["input_directory"]
	bebop_results = json_data["results_directory"]

	results_dir = '{}/results_{}'.format(bebop_results, timestamp)
	if not os.path.exists(results_dir):
		os.mkdir(results_dir)

	results_file = bebop_parsl.submit_cplex_postprocess(bebop_input, results_dir, timestamp)
	#results_fn = os.path.basename(results_file)
	#final_results_file = bebop_results + '/' + results_fn
	#copyfile(results_file, final_results_file)
	print("Copied results to '{}'".format(results_file))


