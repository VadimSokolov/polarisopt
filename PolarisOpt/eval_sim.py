import os
import sys
from subprocess import Popen
import platform
import shutil
import numpy as np
from PolarisOpt.utils import archiver
import sqlite3
import json
from PolarisOpt.utils.objective_funcs import run_objective
import time
from pathlib import Path
import PolarisOpt.slurm_wrappers


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def query_db(db_path, SQL_query):
    db = sqlite3.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute(SQL_query)
    output = cur.fetchall()
    db.close()
    return np.asarray(output)

def copy_simulation(src_dir, dst_dir):
    names = os.listdir(src_dir)
    os.makedirs(dst_dir)
    for name in names:
        src_name = os.path.join(src_dir, name)
        if not os.path.isdir(src_name):
            dst_name = os.path.join(dst_dir, name)
            shutil.copy2(src_name, dst_name)


def run_task(manager, inputs, task):
    r"""Evaluates a set of inputs in the simulator

    Args:
        manager (class): object containing settings for simulation
        inputs (n-array): the new values to be run
        task (int): counter indicating simulation instance being performed

    Returns:
        the single-output objective function and uncollapsed distance from target based on the outputs
    """
    print("Running task: {} - {}".format(task, inputs), flush=True)
    start = time.perf_counter()
    task_dir = os.path.join(manager.working_dir, 'experiments', "Sim"+str(task))
    src_dir = manager.simulation_path
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    
    copy_simulation(src_dir, task_dir)
    # shutil.copytree(src_dir, task_dir)

    archiver.update_json(manager.vnames, inputs, task_dir)

    if hasattr(manager, 'polaris_executable'):
        exe_fp = manager.polaris_executable
    else:
        if platform.system().lower() == 'windows':
            exe_fp = os.path.join(manager.polaris_executable, 'Integrated_model.exe')
        else:
            exe_fp = os.path.join(manager.polaris_executable, 'Integrated_Model')
    
    print('Polaris Executable: {}'.format(exe_fp), flush=True)

    sc_fp = os.path.join(task_dir, manager.simulation_scenario_name)
    if manager.convergence:
       cv_fp=manager.convergence_path
    else:
       cv_fp=None
    print('Polaris Convergence: {}'.format(cv_fp), flush=True)
    if manager.dictionary["slurm"]["useslurm"]:
        slurm_wrappers.eval_sim(task_dir, exe_fp, sc_fp, manager.working_dir, cv_fp)
    else:
        run_sim(task_dir, exe_fp, sc_fp, manager.working_dir, cv_fp)
    print(os.getcwd())
    obj, y_err = pull_result(task_dir, manager)
    end = time.perf_counter()
    return obj, y_err, convert_time(end-start)


def run_sim(task_dir, exe_fp, sc_fp, working_dir, cv_fp=None):
    r"""runs the POLARIS executable with the associated scenario file
    Args:
        task_dir: (path) path to the folder containing the scenario and variable-related files
        exe_fp: (path) full path to the Polaris Integrated_model.exe
        sc_fp: (path) scenario filename within task_dir for the .json needed to run an instance of the executable
        cv_fp: (path) convergence directory path if convergence should be run
    """
    os.chdir(task_dir)
    # Run the Polaris exe file via pipe
    num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'
#    if platform.system().lower() == 'windows':
#        p1 = Popen([exe_fp, sc_fp, num_threads], shell=True)
#    else:
#        num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'

    # create an init scenario file for local task_dir from scenario.json file
    if cv_fp is not None:
        control_fp=create_conv_files(sc_fp, exe_fp, cv_fp, num_threads)
        p1= Popen(['python', os.path.join(cv_fp,'run_convergence.py'),control_fp,task_dir],shell=False)
    else:
        p1 = Popen([exe_fp, sc_fp, num_threads], shell=False)
    p1.wait()
    os.chdir(working_dir)
    return print("task completed")


def create_conv_files(sc_fp,exe_fp, cv_fp, num_threads):
    #create an init version of scenario
    p,f= sc_fp.split(os.extsep) 
    sc_fp_init = p + "_init."+f
    output_base, database_base = pull_basenames(sc_fp)
    output_control_fp = os.path.join(os.path.dirname(sc_fp),database_base + "_control.json")
    dictionary = json.loads(open(sc_fp).read())
    #check with randy
    dictionary["Routing and skimming controls"]["time_dependent_routing"]=False
    dictionary["Population synthesizer controls"]["read_population_from_database"]=False
    dictionary["Population synthesizer controls"]["percent_to_synthesize"]=0.01
    dictionary["Population synthesizer controls"]["demand_reduction_factor"]=0
    with open(sc_fp_init, 'w+') as fp:
        json.dump(dictionary, fp, indent = 4)

    #TO DO: right now num_abm_runs set to 3 by default
    dictionary = {}
    dictionary["scenario_main_init"] = sc_fp_init
    dictionary["scenario_main"] = sc_fp
    dictionary["num_abm_runs"] = 3
    dictionary["output_directories"] = output_base
    dictionary["num_threads"] = num_threads
    dictionary["database_base_name"] = database_base
    dictionary["model"] = exe_fp
    dictionary["scripts_dir"] = cv_fp
    #dictionary["standard_dir"]= 
    dictionary["results_dir"]= "convergence_results"
    with open(output_control_fp, 'w+') as fp:
        json.dump(dictionary, fp, indent = 4)
    return output_control_fp

def pull_basenames(sc_fp):
        dictionary = json.loads(open(sc_fp).read())
        output_base = dictionary['Output controls']['output_dir_name']
        database_base = dictionary["General simulation controls"]['database_name']
        #if platform.system().lower() == 'linux':
        #    output_base = 'linux_{}'.format(output_base)
        return output_base, database_base
        
def pull_result(task_dir, manager):
    r"""Performs a SQL query to retrieve the results and calculates the objective value
    Args:
        task_dir (path): folder path containing the sample-instance files
        manager (class): object containing settings for simulation
    Returns:
        the uncollapsed distance from the target outputs and objective
        value
    """
    task_db = os.path.join(
        task_dir,
        manager.target_output_filename
    )
    target_db = manager._target_output_filepath

    new_output = query_db(task_db, manager.output_SQL_query)
    ref_output = query_db(target_db, manager.output_SQL_query)

    if new_output.shape[0] != ref_output.shape[0]:
        print('output mismatch by {} and {}'.format(new_output.shape, ref_output.shape))
        obj = "P"
        er = None
        return obj, er
    else:
        return run_objective(ref_output[:, 1]-new_output[:, 1], manager.objective_type)


def eval_sample_task(manager, output_fp, inputs, task, write_record=True):
    r"""Evaluates a set of inputs generated in the original subspace and records the outcome

    Args:
        manager (class): object containing settings for simulation
        output_fp (path): full path to file outputs should be saved to
        inputs (n-array): the new values to be run
        task (int): counter indicating simulation instance being performed

    Returns:
        a results file containing the target error and objective values for the run
    """
    obj, y_err, rtime = run_task(manager, inputs, task)
    # print(f'{type(obj)}, {type(y_err)}, {type(rtime)}', flush=True)
    if write_record:
        update_sample_record(obj, y_err, rtime, output_fp, inputs)
    return (obj, y_err, rtime, task)

def update_sample_record(obj, y_err, rtime, output_fp, inputs):
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
            ["status", "objective", "target_err", "run_time"],
            [["Completed", obj, y_err, rtime]],
            output_fp,
            identifier_key="orig_input"
        )


def eval_DR_task(manager, DR_model, DR_input, task, write_record=True):
    r"""Evaluates a set of inputs and records the outcome

    Args:
        manager (class): object containing settings for simulation
            simulation_path: path to the folder containing the simulation
        DR_model (class): the model using the Dimension Reduction
        DR_input (n-array): the new values
        task (int): counter indicating simulation instance being performed

    Returns:
        a results file containing the target error and objective functions for the task
    """
    print("running sample %d \n" % task)

    xhat = DR_model.decode_X(DR_input)
    obj, y_err, rtime = run_task(manager, xhat, task)
    if write_record:
        update_DR_record(obj, y_err, rtime, DR_input, xhat, manager)
    return (obj, y_err, rtime, xhat, task)


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
