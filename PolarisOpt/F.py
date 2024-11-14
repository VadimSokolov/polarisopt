"""
    This file contains the main function calls
"""

import os, sys
import numpy as np
from concurrent import futures
from itertools import repeat
import time
import json
from PolarisOpt import custom_gp as cgp
from PolarisOpt.utils import sampler
from PolarisOpt.utils import archiver
from PolarisOpt.utils import util
from PolarisOpt import eval_sim
from PolarisOpt import bo
from PolarisOpt import dim_red
import shutil

def copy_simulation(src_dir, dst_dir):
    names = os.listdir(src_dir)
    for name in names:
        src_name = os.path.join(src_dir, name)
        if not os.path.isdir(src_name):
            dst_name = os.path.join(dst_dir, name)
            shutil.copy2(src_name, dst_name)

def create_simulation_folder(task, manager):
    if not os.path.exists(task.task_dir):
        os.makedirs(task.task_dir)    
        print(f'Copying simulation files to {task.task_dir}', flush=True)
        copy_simulation(manager.simulation_path, task.task_dir)
    archiver.update_json(manager.vnames, task.sample, task.task_dir) # replace the json file with the new inputs

class SampleTask:
    def __init__(self, task_dir, var_values,run_id):
        self.task_dir = task_dir
        self.sample = var_values
        self.run_id = run_id
        self.obj=None
        self.y_err = None
        self.rtime = None
        self.complete = False
        self.start_iteration_from = 1

def create_task(run_id,inputs,manager):
    task_dir = os.path.join(manager.working_dir, 'experiments', "Sim"+str(run_id))
    # check if task was already executed
    scenariopath = os.path.join(task_dir, manager.polaris_scenario_file)
    if manager.convergence:
        lastit, _ = manager.check_iterations(task_dir,scenariopath)
        if lastit >= manager.num_abm_runs:
            print(f"Simulation {task_dir} was already run {manager.num_abm_runs} times. Not adding this task to the queue.")
            return None
    else:
        if os.path.exists(scenariopath):
            task_output = manager.get_task_output(task_dir,scenariopath)
            if os.path.exists(os.path.join(task_output,'finished')):
                print(f"Finished file was created in {task_output}. Not adding this task to the queue.")
                return None
    task = SampleTask(task_dir, inputs,run_id)
    if manager.convergence:
        task.start_iteration_from  = lastit
    return task



def evaluate_samples(manager, max_parallel = 2, num_samples = 0, eq_sql=None):
    r"""Function which runs all necessary steps to (create and) evaluate a sample training file.
    Args:
        manager (SetupManager class): central parameter keeper
        max_parallel (int): the largest number of parallel evaluations allowed while evaluating all pending samples
                in the manager.training_filename file
        num_samples (int): the number of samples taken from a Lating Hypercube constructed across the statespace
                If num_samples = 0, no additional samples will be created
        
    Returns:
      a file containing the evaluated samples in the format necessary for training [Y, X]
    """

    # manager.training_filename (text): the file name to place evaluated or pending points into. Will be placed
    # in the 'data' folder and in the format of [Y,X]
    
    training_file = manager._check_file(manager.training_filename)

    #################################
    #STEP 1: Create LHS if desired  #
    #################################
    if num_samples>0:
        pend_samples = sampler.LHS_pool(manager.orig_range[0], num_samples, manager.orig_range[1])
        archiver.create_record(pend_samples, training_file, var_names = manager.var, identifier_key = "orig_input")
    else:
        _, pend_samples = archiver.import_dataset(training_file, x_key = "orig_input", y_key = "target_err")
    n = len(pend_samples)
    task_ids = range(manager.run_id,manager.run_id + n)
    manager.run_id+=n
    tasks = []
    for i in range(n):
        run_id = task_ids[i]
        inputs = pend_samples[i]
        task = create_task(run_id,inputs,manager)
        if task is not None:
            tasks.append(task)
            
    with futures.ThreadPoolExecutor(max_parallel) as executor:
        result = executor.map(eval_sim.run_task, repeat(manager), tasks)
        # result = executor.map(eval_sim.eval_sample_task_mock, repeat(manager), tasks)
        for task in result:
            if not task.complete: # Execution failed
                print(f'Error evaluating sample {task.run_id}, skipping....')
            else:
                eval_sim.update_sample_record(task.obj, task.y_err, task.rtime, training_file, task.sample,task.task_dir)




def build_calibration(manager, quiet = True):
    r"""Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
    Args:
        manager (SetupManager class): central parameter keeper
        quiet (boolean): to (False) or not to (True) print training progress        
    Returns:
        DR_model (DR_Technique class): class containing dimension reduction technique
        M_model (Mean_NN class): class containing the GP NN mean if implemented
    """

    ########################
    # Setup and Train model#
    ########################

    DR_model = dim_red.create_DR(manager, quiet = quiet)

    if manager.add_nn_GP_mean:
        _, _, _, _, NN_mean_var = archiver.load_DR_settings(manager._settings_filepath)
        M_model = cgp.Mean_NN([manager.dim_in, manager.dim_out, *NN_mean_var], DR_model)
        M_model.calculate(manager, quiet = quiet)
        #auto saves to data folder
        model_fp = os.path.join(manager.model_dir, 'mean_model.pickle')
        archiver.save_model(M_model, model_fp)
    else:
        M_model = None
    return DR_model, M_model        



def calibrate_simulation(manager, DR_model, M_model = None, max_parallel=2, quiet=True, eq_sql=None):
    r"""Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
    Args:
        manager (SetupManager class): central parameter keeper
        DR_model (DR_Technique class): class containing dimension reduction technique
        M_model (Mean_NN class): class containing the GP NN mean if implemented
        quiet (boolean): to (False) or not to (True) print additional progress        
        
    Returns:
      a .pickle file containing the dimension-reduction model
      a .pickle file containing the mean model, if applicable
      a .json file containing the Bayes Opt results 
    """

    DR_updates, mean_updates = archiver.load_update_settings(manager._settings_filepath)
        
    # going to have to run 1 more than number of requested loops to record the final loop's returned data points
    for l in range(0, manager.num_BO_loops+1):
        #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
        eval_samples, pend_samples = manager.load_results()
        if abs(min(eval_samples[:,0])) > manager.epsilon_stop:
            if eq_sql is not None:
                print("EQSQL", flush=True)
                from eqsql import proxies
                from eqsql import eq
                import eval_wrapper
                func = proxies.dump_proxies(f=eval_wrapper.eval_dr_task)['f']
                # print(manager, DR_model, pend_samples)
                proxy_js = proxies.dump_proxies(manager=manager, dr_model=DR_model,
                                                pend_samples=pend_samples)
                exp_id = os.getenv("EXP_ID")
                payload = {'func': func, 'proxies': proxy_js, 'parameters': [{'row': r} for r in range(len(pend_samples))]}
                status, ft = eq_sql.submit_task(exp_id, eq_type=0, payload=json.dumps(payload))
                if status != eq.ResultStatus.SUCCESS:
                    eq_sql.stop_worker_pool(eq_type=0)
                    raise ValueError("Error submitting task while attempting to calibrate simulation")
                # timeout should be set to max duration of polaris run in seconds
                timeout = float(os.getenv("ME_TIMEOUT"))
                status, result = ft.result(timeout=timeout)
                if status != eq.ResultStatus.SUCCESS:
                    # don't call this as it typically leaves a stop flag in the DB
                    # for the next run
                    # eq.stop_worker_pool(eq_type=0)
                    raise ValueError("Error querying task result while attempting to calibrate simulation: {}".format(result))
                
                result_dict = json.loads(result)
                proxy_result = proxies.load_proxies(result_dict['proxies'])
                result_list = proxy_result['results']
                # print(f'RESULT: {result_list}', flush=True)
                for obj, y_err, rtime, xhat, task_id in result_list:
                    eval_sim.update_DR_record(obj, y_err, rtime, pend_samples[task_id], xhat, manager)
            else:
                with futures.ThreadPoolExecutor(max_parallel) as executor:
                    result = executor.map(eval_sim.eval_DR_task, repeat(manager), repeat(DR_model), pend_samples, 
                                        (x for x in range(len(pend_samples))), repeat(False))
                    for obj, y_err, rtime, xhat, task_id in result:
                        eval_sim.update_DR_record(obj, y_err, rtime, pend_samples[task_id], xhat, manager)
               # util.thread_it(eval_sim.eval_DR_task, [(manager, DR_model, pend_samples[row], row) for row in range(len(pend_samples))])

        if DR_updates[0]:
            #TODO: this currently wipes out any pending recommended samples when updating
            if (l+1) % DR_updates[1] == 0:
                DR_model = dim_red.tune_DR(manager, quiet = quiet)
                time.sleep(10)
        if manager.add_nn_GP_mean and mean_updates[0]:
            if (l+1) % mean_updates[1] == 0:
                #need to re-arrange the results file as a new 'training file'
                M_model.tune(manager, quiet = quiet)
                model_fp = os.path.join(manager.model_dir, 'mean_model.pickle')
                archiver.save_model(M_model, model_fp)
                time.sleep(10)

        if l<manager.num_BO_loops:
            #If less then the number of trials we run, run another Bayes set
            print("running loop number %d of %d" % (l+1, manager.num_BO_loops))
            bo.main_loop(manager, DR_model = DR_model, M_model = M_model)

    print("Review %s file in data directory"% manager.res_filename)        

