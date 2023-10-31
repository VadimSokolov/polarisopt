import os
import sys
from subprocess import Popen
import platform

import numpy as np
import h5py
from PolarisOpt.utils import archiver
import sqlite3
import json
from PolarisOpt.utils.objective_funcs import run_objective
import time
from pathlib import Path
from PolarisOpt.slurm_wrappers import run_sim_slurm
from PolarisOpt.F import create_simulation_folder


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def query_db(db_path, SQL_query):
    db = sqlite3.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute(SQL_query)
    output = cur.fetchall()
    db.close()
    return np.asarray(output)


def run_sim_local(task_dir, polarisbin, scenariopath, working_dir, convrgencepath=None):
    r"""runs the POLARIS executable with the associated scenario file
    Args:
        task_dir: (path) path to the folder containing the scenario and variable-related files
        polarisbin: (path) full path to the Polaris Integrated_model.exe
        scenariopath: (path) scenario filename within task_dir for the .json needed to run an instance of the executable
        convrgencepath: (path) convergence directory path if convergence should be run
    """
    os.chdir(task_dir)
    # Run the Polaris exe file via pipe
    num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'
#    if platform.system().lower() == 'windows':
#        p1 = Popen([polarisbin, scenariopath, num_threads], shell=True)
#    else:
#        num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'

    # create an init scenario file for local task_dir from scenario.json file
    if convrgencepath is not None:
        control_fp=create_conv_files(scenariopath, polarisbin, convrgencepath, num_threads)
        p1= Popen(['python', os.path.join(convrgencepath,'run_convergence.py'),control_fp,task_dir],shell=False)
    else:
        p1 = Popen([polarisbin, scenariopath, num_threads], shell=False)
    p1.wait()
    os.chdir(working_dir)
    return print("Task completed")

def create_conv_files(scenariopath,polarisbin, convrgencepath, num_threads):
    #create an init version of scenario
    p,f= scenariopath.split(os.extsep) 
    scenariopath_init = p + "_init."+f
    output_base, database_base = pull_basenames(scenariopath)
    output_control_fp = os.path.join(os.path.dirname(scenariopath),database_base + "_control.json")
    dictionary = json.loads(open(scenariopath).read())
    #check with randy
    dictionary["Routing and skimming controls"]["time_dependent_routing"]=False
    dictionary["Population synthesizer controls"]["read_population_from_database"]=False
    dictionary["Population synthesizer controls"]["percent_to_synthesize"]=0.01
    dictionary["Population synthesizer controls"]["demand_reduction_factor"]=0
    with open(scenariopath_init, 'w+') as fp:
        json.dump(dictionary, fp, indent = 4)

    #TO DO: right now num_abm_runs set to 3 by default
    dictionary = {}
    dictionary["scenario_main_init"] = scenariopath_init
    dictionary["scenario_main"] = scenariopath
    dictionary["num_abm_runs"] = 3
    dictionary["output_directories"] = output_base
    dictionary["num_threads"] = num_threads
    dictionary["database_base_name"] = database_base
    dictionary["model"] = polarisbin
    dictionary["scripts_dir"] = convrgencepath
    #dictionary["standard_dir"]= 
    dictionary["results_dir"]= "convergence_results"
    with open(output_control_fp, 'w+') as fp:
        json.dump(dictionary, fp, indent = 4)
    return output_control_fp

def pull_basenames(scenariopath):
        dictionary = json.loads(open(scenariopath).read())
        output_base = dictionary['Output controls']['output_dir_name']
        database_base = dictionary["General simulation controls"]['database_name']
        #if platform.system().lower() == 'linux':
        #    output_base = 'linux_{}'.format(output_base)
        return output_base, database_base
        
def pull_result(task_output,manager):
    r"""Performs a SQL query to retrieve the results and calculates the objective value
    Args:
        task_dir (path): folder path containing the sample-instance files
        manager (class): object containing settings for simulation
    Returns:
        the uncollapsed distance from the target outputs and objective
        value
    """
    # task_output = manager.get_task_output(task_dir,scenariopath)
    # task_db = os.path.join(task_output,manager.result_filename)
    # target_db = manager.target_output_filepath
    task_db = os.path.join(task_output,manager.result_filename) 
    target_db = manager.target_output_filepath
    with h5py.File(task_db, 'r') as f:
        new_output = f['link_moe']['link_travel_time'][:]*f['link_moe']['link_in_volume'][:]
        new_output =  np.mean(new_output, axis=1)
    with h5py.File(target_db, 'r') as f:
        ref_output = f['link_moe']['link_travel_time'][:]*f['link_moe']['link_in_volume'][:]
        ref_output =  np.mean(ref_output, axis=1)


    # new_output = query_db(task_db, manager.output_SQL_query)
    # ref_output = query_db(target_db, manager.output_SQL_query)

    if new_output.shape[0] != ref_output.shape[0]:
        print('output mismatch by {} and {}'.format(new_output.shape, ref_output.shape))
        obj = "P"
        er = None
        return obj, er
    else:
        return run_objective(ref_output - new_output, manager.objective_type)

def run_task(manager, task):
    r"""Runs the simulator from task_dir

    Args:
        manager (class): object containing settings for simulation
        task (SampleTask): contains necessary information about the task to be executed

    Returns:
        the single-output objective function and uncollapsed distance from target based on the outputs
    """
    # print(f'Running run_id: {task.run_id}', flush=True)
    start = time.perf_counter()
    if hasattr(manager, 'polaris_executable'):
        polarisbin = manager.polaris_executable
    else:
        if platform.system().lower() == 'windows':
            polarisbin = os.path.join(manager.polaris_executable, 'Integrated_model.exe')
        else:
            polarisbin = os.path.join(manager.polaris_executable, 'Integrated_Model')
    
    # print('Polaris Executable: {}'.format(polarisbin), flush=True)
    create_simulation_folder(task,manager)
    scenariopath = os.path.join(task.task_dir, manager.simulation_scenario_name)
    task_output = manager.get_task_output(task.task_dir,scenariopath)
    if manager.convergence:
       convrgencepath=manager.convergence_path
    else:
       convrgencepath=None
    # print('Polaris Convergence: {}'.format(convrgencepath), flush=True)
    # print(f'Using slurm flag: {manager.dictionary["slurm"]["useslurm"]}', flush=True)
    if manager.dictionary["slurm"]["useslurm"]:
        # print(f'Submitting the slurm job: {task.task_dir}', flush=True)
        # res = run_sim_slurm(task, polarisbin, scenariopath, convrgencepath,manager,task.run_id)
        res = run_sim_slurm(task,polarisbin,scenariopath,manager)
        if res is False:
            return task
        else:
            print(res)
    else:
        print('Running the simulation locally', flush=True)
        run_sim_local(task.task_dir, polarisbin, scenariopath, manager.working_dir, convrgencepath)
    task.obj, task.y_err = pull_result(task_output,manager)
    end = time.perf_counter()
    task.rtime = convert_time(end-start)
    task.completed = True
    return task

def update_sample_record(obj, y_err, rtime, output_fp, inputs,tasks_dir=None):
    if obj == "P":
        archiver.update_record(
            [inputs],
            ["status", "run_time"],
            [["Errored", rtime]],
            output_fp,
            identifier_key="orig_input"
        )
    else:
        archiver.update_record(
            [inputs],
            ["status", "objective", "target_err", "run_time","tasks_direcotry"],
            [["Completed", obj, y_err, rtime,tasks_dir]],
            output_fp,
            identifier_key="orig_input"
        )


def eval_DR_task(manager, DR_model, DR_input, run_id, write_record=True):
    r"""Evaluates a set of inputs and records the outcome

    Args:
        manager (class): object containing settings for simulation
            simulation_path: path to the folder containing the simulation
        DR_model (class): the model using the Dimension Reduction
        DR_input (n-array): the new values
        run_id (int): counter indicating simulation instance being performed

    Returns:
        a results file containing the target error and objective functions for the task
    """
    print("running sample %d \n" % run_id)

    xhat = DR_model.decode_X(DR_input)
    obj, y_err, rtime, task_dir = run_task(manager, xhat, run_id)
    if write_record:
        update_DR_record(obj, y_err, rtime, DR_input, xhat, manager)
    return (obj, y_err, rtime, xhat, run_id)


def update_DR_record(obj, y_err, rtime, DR_input, xhat, manager):
    if obj == "P":
        archiver.update_record(
            [DR_input],
            ["status", "orig_input", "run_time"],
            [["Errored", xhat, rtime]],
            manager._res_filepath,
            identifier_key="DR_input"
        )
    else:
        archiver.update_record(
            [DR_input],
            ["status", "orig_input", "objective", "target_err", "run_time"],
            [["Completed", xhat, obj, y_err, rtime]],
            manager._res_filepath,
            identifier_key="DR_input"
        )
