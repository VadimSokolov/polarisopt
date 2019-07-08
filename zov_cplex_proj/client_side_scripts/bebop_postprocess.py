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
	results_dir = sys.argv[1]
	timestamp = sys.argv[2]

	if not os.path.exists(results_dir):
		os.mkdir(results_dir)

	results_file = bebop_parsl.submit_cplex_postprocess(results_dir, timestamp)
	print("Copied results to '{}'".format(results_dir))


