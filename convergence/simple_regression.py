#!/usr/bin/python
# Filename: simpl_regression.py

import sys
import os
import subprocess
from shutil import copyfile
import regression

if (len(sys.argv) < 3):
	print(('Usage: %s <standard_directory> <simulated_directory>') % (str(sys.argv[0])))
	sys.exit(-1)


standard_dir = sys.argv[1]
simulated_dir = sys.argv[2]

print(('Running regression on \'%s\'') % (str(simulated_dir)))
regression.regression(standard_dir, simulated_dir)

