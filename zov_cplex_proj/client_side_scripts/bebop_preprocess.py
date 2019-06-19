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

	results_file = bebop_parsl.submit_cplex_preprocess(timestamp)


