#!/usr/bin/python
# Filename: run_regression.py

import shutil
from pathlib import Path
import sys
import os
import subprocess
from shutil import copyfile
import json
import regression

if (len(sys.argv) < 2):
	print(('Usage %s <json_control_file>') % (sys.argv[0]))
	sys.exit(-1)

with open(sys.argv[1]) as f:
  json_data = json.load(f)

print(json_data)

#exe_name = sys.argv[1]
#results_dir = sys.argv[2]

exe_name = json_data["model"]
data_dir = json_data["data"]
results_dir = data_dir + '/' + json_data["results_dir"]
scenario_file = json_data["model_scenario"]
num_threads = json_data["num_threads"]
standard_dir = json_data["standard_dir"]

cwd = os.getcwd()
os.chdir(data_dir)

# Create results Directory if don't exist
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    print("Directory " , results_dir ,  " Created ")
else:    
    print("Directory " , results_dir ,  " already exists")

arguments = scenario_file + ' ' + num_threads
print('Executing \'' + exe_name + ' ' + arguments + '\'')

#subprocess.call([exeName, arguments])
out_file = open(results_dir + '/simulation_out.log', 'w+')
err_file = open(results_dir + '/simulation_err.log', 'w+')
proc = subprocess.Popen([exe_name, scenario_file, num_threads],
                        stdout=out_file, stderr=subprocess.PIPE)
for line in proc.stderr:
    sys.stdout.write(str(line))
    err_file.write(str(line))
proc.wait()
out_file.close()
err_file.close()

all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
latest_subdir = max(all_subdirs, key=os.path.getmtime)

#standard_dir = 'Regression_test'
simulated_dir = latest_subdir

#move the output files (now that we know where the simulation files were created)
shutil.move(results_dir, simulated_dir + '/' + json_data["results_dir"])
#os.rename('./simulation_out.log', simulated_dir + '/simulation_out.log')
#os.rename('./simulation_err.log', simulated_dir + '/simulation_err.log')

print('Running regression on \'' + simulated_dir + '\'')
regression.regression(standard_dir, simulated_dir)

##make sure results_dir exists
#if not os.path.exists(results_dir):
#    os.makedirs(results_dir)
	
#copyfile('./Regression_Report.html', results_dir + '/Regression_Report.html')
#copyfile(simulated_dir + '/in_network.png', results_dir + '/in_network.png')
#copyfile(simulated_dir + '/simulation_out.log', results_dir + '/simulation_out.log')
#copyfile(simulated_dir + '/simulation_err.log', results_dir + '/simulation_err.log')
