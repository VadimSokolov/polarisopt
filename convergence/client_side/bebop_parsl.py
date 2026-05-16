# from parsl.app.app import python_app
from parsl.channels import SSHChannel
from parsl.providers import SlurmProvider
# from parsl.providers import LocalProvider
# from parsl.channels import LocalChannel
from parsl.providers.slurm.slurm import translate_table
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SrunLauncher
# import parsl
# from parsl.configs import local_threads
# from parsl.configs.local_threads import config
import os
from os import walk

import time
# import random
# import uuid
# import subprocess

# Script for testing remote job submission to Bebop.  This script starts a remote
# job that adds two numbers together in a shell script, writes the result to a
# file, and then returns the result file to the client.

# !!!! IMPORTANT - Do NOT write to user HOME directory.  Use a sub-dir or better
#                  yet a project folder, otherwise the user home folder permissions
#                  can be changed and future login will fail !!!!!


# preprocess not needed
#
def submit_convergence_preprocess(timestamp, input_directory):
	# Remote working directory.  
	wd = '/lcrc/project/POLARIS/bebop/scratch'

	# Define the SLURM job properties, such as batch queue name, wall time, etc.
	provider = SlurmProvider(
		'knlall',
		launcher=SingleNodeLauncher(),
		scheduler_options='',     # Input your scheduler_options if needed
		worker_init='cd {}'.format(wd),     # Input your worker_init if needed
		walltime="00:10:00",
		init_blocks=1,
		max_blocks=1,
		nodes_per_block=1,
		channel=SSHChannel('bebop.lcrc.anl.gov', username='rweimer', script_dir='/lcrc/project/POLARIS/bebop/scratch'),
	)

	# this has to be set to an existing directory otherwise provides tries to write
	# submission script to 'None'
	provider.script_dir = '.'
	completed_tag = translate_table['CD']

	pre_job_ids = []
	preprocess_cmd = '{}/cplex_preprocess.sh {}'.format(wd, timestamp)
	pre_job_ids.append(provider.submit(preprocess_cmd, 1, 1))
	print("PREPROCESS JOB ID: {}".format(pre_job_ids[0]))
	pre_status = provider.status(pre_job_ids)
	print('PREPROCESS JOB STATUS: {}'.format(pre_status))

	# Wait for preprocess job to complete
	while all(stat != "COMPLETED" for stat in pre_status):
		time.sleep(10)
		pre_status = provider.status(pre_job_ids)
		print('PREPROCESS JOB STATUS: {}'.format(pre_status))


def submit_convergence_postprocess(results_directory, timestamp):
	# Remote working directory.  
	wd = '/lcrc/project/POLARIS/bebop/scratch'

	# Define the SLURM job properties, such as batch queue name, wall time, etc.
	provider = SlurmProvider(
		'knlall',
		launcher=SingleNodeLauncher(),
		scheduler_options='',     # Input your scheduler_options if needed
		worker_init='cd {}'.format(wd),     # Input your worker_init if needed
		walltime="01:00:00",
		init_blocks=1,
		max_blocks=1,
		nodes_per_block=1,
		channel=SSHChannel('bebop.lcrc.anl.gov', username='rweimer', script_dir='/lcrc/project/POLARIS/bebop/scratch')
	)

	# this has to be set to an existing directory otherwise provides tries to write
	# submission script to 'None'
	provider.script_dir = '.'
	completed_tag = translate_table['CD']

	# Define the remote job result output file
	fname = 'result_{}.txt'.format(timestamp)
	fpath = '{}/run_{}/results/{}'.format(wd, timestamp, fname)

	ppc_job_ids = []
	post_process_cmd = '{}/cplex_postprocess.sh {} {}'.format(wd, fname, timestamp)
	ppc_job_ids.append(provider.submit(post_process_cmd, 1, 1))

	print("POSTPROCESS JOB ID: {}".format(ppc_job_ids[0]))
	completed_tag = translate_table['CD']
	ppc_status = provider.status(ppc_job_ids)
	print('POSTPROCESS JOB STATUS: {}'.format(ppc_status))

	# Wait for post job to complete
	while all(stat != completed_tag for stat in ppc_status):
		time.sleep(10)
		ppc_status = provider.status(ppc_job_ids)
		print('POSTPROCESS JOB STATUS: {}'.format(ppc_status))

	# Get the result file from the remote machine
	list_of_results_file = '{}/run_{}/results/all_results.txt'.format(wd, timestamp)
	provider.channel.pull_file(list_of_results_file, results_directory)

	results_list = '{}/all_results.txt'.format(results_directory)
	file_list = [line.rstrip('\n') for line in open(results_list)]
	for name in file_list:
		# remote_filename = '{}/run_{}/results/{}'.format(wd, timestamp, name)
		print('Pulling file: \'{}\' to : \'{}\''.format(name, results_directory))
		provider.channel.pull_file(name, results_directory)

	provider.channel.close()
	os.remove('{}/all_results.txt'.format(results_directory))


def submit_convergence_jobs(input_directory, results_directory, timestamp):
	# Remote working directory.  
	wd = '/lcrc/project/POLARIS/bebop/scratch'

	# Define the SLURM job properties, such as batch queue name, wall time, etc.
	provider = SlurmProvider(
		'knlall',
		launcher=SrunLauncher(),
		scheduler_options='',     # Input your scheduler_options if needed
		worker_init='cd {}'.format(wd),     # Input your worker_init if needed
		walltime="01:00:00",
		init_blocks=1,
		max_blocks=1,
		nodes_per_block=1,
		channel=SSHChannel('bebop.lcrc.anl.gov', username='rweimer', script_dir='/lcrc/project/POLARIS/bebop/scratch')
	)

	# this has to be set to an existing directory otherwise provides tries to write
	# submission script to 'None'
	provider.script_dir = '.'
	completed_tag = translate_table['CD']

	# push the input file
	# Submit the job
	# command, block size, tasks per node,
	job_ids = []
	print('Transferring input files from {} and submitting jobs...'.format(input_directory))
	for dir_path, dirnames, filenames in walk(input_directory):
		for file in filenames:
			local_input_file = input_directory + '/' + file
			remote_input_file = '{}/run_{}/inputs/{}'.format(wd, timestamp, file)
			remote_output_file = '{}/run_{}/outputs/results_{}'.format(wd, timestamp, file)
			print('Pushing file: \'{}\' to : \'{}\''.format(local_input_file, remote_input_file))
			provider.channel.push_file(local_input_file, '{}/run_{}/inputs'.format(wd, timestamp))
			output_file = 'output_' + file
			cplex_cmd = '{}/cplex_ihvs {} {}'.format(wd, remote_input_file, remote_output_file)
			print('Submitting Job: \'{}\''.format(cplex_cmd))
			job_ids.append(provider.submit(cplex_cmd, 1, 64))

	completed_tag = translate_table['CD']
	status = provider.status(job_ids)

	# Wait for job to complete
	while not all(stat == "COMPLETED" for stat in status):
		time.sleep(10)
		status = provider.status(job_ids)
		print('JOB STATUS: waiting...{}'.format(status))
	job_ids[:] = []
