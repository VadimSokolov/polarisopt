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


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def query_db(db_path, SQL_query):
    db = sqlite3.connect(db_path, timeout=120)
    cur = db.cursor()
    cur.execute(SQL_query)
    output = cur.fetchall()
    db.close()
    return np.asarray(output)


def run_task(manager, inputs, task):
    r"""Evaluates a set of inputs in the simulator

    Args:
        manager (class): object containing settings for simulation
        inputs (n-array): the new values to be run
        task (int): counter indicating simulation instance being performed

    Returns:
        the single-output objective function and uncollapsed distance from target based on the outputs
    """
    start = time.perf_counter()
    task_dir = os.path.join(manager.working_dir, 'experiments', "Sim"+str(task))
    src_dir = manager.simulation_path
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    shutil.copytree(src_dir, task_dir)

    archiver.update_json(manager.vnames, inputs, task_dir)

    if platform.system().lower() == 'windows':
        exe_fn = os.path.join(manager.simulation_path, 'Integrated_model.exe')
    else:
        exe_fn = os.path.join(manager.simulation_path, 'Integrated_Model')

    run_sim(task_dir, exe_fn, manager.simulation_scenario_name,manager.working_dir)
    print(os.getcwd())
    obj, y_err = pull_result(task_dir, manager)
    end = time.perf_counter()
    return obj, y_err, convert_time(end-start)


def run_sim(task_dir, exe_fn, sc_fn, working_dir):
    r"""runs the POLARIS executable with the associated scenario file
    Args:
        task_dir: (path) path to the folder containing the scenario and variable-related files
        exe_fn: (path) full path to the Polaris Integrated_model.exe
        sc_fn: (path) scenario filename within task_dir for the .json needed to run an instance of the executable
    """
    os.chdir(task_dir)
    # Run the Polaris exe file via pipe
    num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'
#    if platform.system().lower() == 'windows':
#        p1 = Popen([exe_fn, sc_fn,num_threads], shell=True)
#    else:
#        num_threads = os.environ['POLARIS_NUM_THREADS'] if 'POLARIS_NUM_THREADS' in os.environ else '1'
    p1 = Popen([exe_fn, sc_fn, num_threads], shell=False)
    p1.wait()
    os.chdir(working_dir)
    return print("task completed")


def pull_result(task_dir, manager):
    r"""Performs a SQL query to retrieve the results and calculates the objective value
    Args:
        task_dir (path): folder path containing the sample-instance files
        manager (class): object containing settings for simulation
            simulation_path: path to the folder containing the simulation
    Returns:
        the uncollapsed distance from the target outputs and objective
        value
    """
    sc_fn = os.path.join(task_dir, manager.simulation_scenario_name)
    dictionary = json.loads(open(sc_fn).read())
    output_dir = dictionary['Output controls']['output_dir_name']
    if platform.system().lower() == 'linux':
        output_dir = 'linux_{}'.format(output_dir)
    print(output_dir, flush=True)
    task_db = os.path.join(
        task_dir,
        output_dir,
        os.path.split(manager.target_output_filename)[1]
    )
    target_db = manager.target_output_filename

    new_output = query_db(task_db, manager.output_SQL_query)
    ref_output = query_db(target_db, manager.output_SQL_query)

    if new_output.shape[0] != ref_output.shape[0]:
        print('output mismatch by {} and {}'.format(new_output.shape, ref_output.shape))
        obj = "P"
        er = None
        return obj, er
    else:
        return run_objective(ref_output[:, 1]-new_output[:, 1], manager.objective_type)


def eval_sample_task(manager, output_fn, inputs, task):
    r"""Evaluates a set of inputs generated in the original subspace and records the outcome

    Args:
        manager (class): object containing settings for simulation
            simulation_path: path to the folder containing the simulation
        output_fn (path): full path to file outputs should be saved to
        inputs (n-array): the new values to be run
        task (int): counter indicating simulation instance being performed

    Returns:
        a results file containing the target error and objective values for the run
    """
    print("running sample %d \n" % task)

    obj, y_err, rtime = run_task(manager, inputs, task)
    # print(f'{obj}, {y_err}, {rtime}', flush=True)
    if obj == "P":
        archiver.update_record(
            [inputs],
            ["status", "run_time"],
            [["Errored", rtime]],
            output_fn,
            identifier_key="orig_input"
        )
    else:
        archiver.update_record(
            [inputs],
            ["status", "objective", "target_err", "run_time"],
            [["Completed", obj, y_err, rtime]],
            output_fn,
            identifier_key="orig_input"
        )


def eval_DR_task(manager, DR_model, DR_input, task):
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
    if obj == "P":
        archiver.update_record(
            [DR_input],
            ["status", "orig_input", "run_time"],
            [["Errored", xhat, rtime]],
            manager.res_filename,
            identifier_key="DR_input"
        )
    else:
        archiver.update_record(
            [DR_input],
            ["status", "orig_input", "objective", "target_err", "run_time"],
            [["Completed", xhat, obj, y_err, rtime]],
            manager.res_filename,
            identifier_key="DR_input"
        )
