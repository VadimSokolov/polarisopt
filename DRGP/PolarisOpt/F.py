"""
    This file contains the main function calls
"""

import os
import time
import json
from concurrent import futures
from itertools import repeat
from PolarisOpt import custom_gp as cgp
from PolarisOpt.utils import sampler
from PolarisOpt.utils import archiver
from PolarisOpt.utils import util
from PolarisOpt import eval_sim
from PolarisOpt import bo
from PolarisOpt import dim_red


def build_sampleset(manager, res_fn, max_parallel=2, num_samples=0, use_emews=False):
    """Function which runs all necessary steps to (create and) evaluate a sample training file.
    Args:
        manager (SetupManager class): central parameter keeper
        res_fn (text): the file name to place evaluated or pending points into. Will be placed
        in the 'data' folder and in the format of [Y,X]
        max_parellel (int): the largest number of parallel evaluations allowed while evaluating all pending samples
                in the res_fn file
        num_samples (int): the number of samples taken from a Lating Hypercube constructed across the statespace
                If num_samples = 0, no additional samples will be created

    Returns:
      a file containing the evaluated samples in the format necessary for training [Y, X]
    """
    res_fn, res_fp = manager._check_file(res_fn)

    #################################
    # STEP 1: Create LHS if desired #
    #################################
    if num_samples > 0:
        pend_samples = sampler.LHS_pool(manager.orig_range[0], num_samples, manager.orig_range[1])
        archiver.create_record(pend_samples, res_fp, var_names=manager.var, identifier_key="orig_input")
    else:
        _, pend_samples = archiver.import_dataset(res_fp, x_key="orig_input", y_key="target_err")

    if use_emews:
        import proxies
        import eq
        import eval_wrapper

        func = proxies.dump_proxies(f=eval_wrapper.eval_sample_task)['f']
        proxy_js = proxies.dump_proxies(manager=manager, output_fp=res_fp,
                                        pend_samples=pend_samples)
        exp_id = os.getenv("EXP_ID")
        payload = {'func': func, 'proxies': proxy_js, 'parameters': [{'row': r} for r in range(len(pend_samples))]}
        status, eq_task_id = eq.submit_task(exp_id, eq_type=0, payload=json.dumps(payload))
        if status != eq.ResultStatus.SUCCESS:
            eq.stop_worker_pool(eq_type=0)
            raise ValueError("Error submitting task while attempting to calibrate simulation")
        # timeout should be set to max duration of polaris run in seconds
        timeout = float(os.getenv("ME_TIMEOUT"))
        status, result = eq.query_result(eq_task_id, timeout=timeout)
        if status != eq.ResultStatus.SUCCESS:
            eq.stop_worker_pool(eq_type=0)
            raise ValueError("Error querying task result while attempting to calibrate simulation: {}".format(result))
        # args = [(manager, res_fp, pend_samples[row], row) for row in range(len(pend_samples))]
        # tmp_dir = os.path.join(os.environ.get("TURBINE_OUTPUT"), 'tmp')
        # pool = emews.Pool(tmp_dir, rank_type="workers")
        # pool.map(eval_sim.eval_sample_task, args)
    else:
        with futures.ThreadPoolExecutor(max_parallel) as executor:
            result = executor.map(eval_sim.eval_sample_task, repeat(manager), repeat(res_fp), pend_samples, 
                                 (x for x in range(len(pend_samples))), repeat(False))
            for obj, y_err, rtime, task_id in result:
                eval_sim.update_sample_record(obj, y_err, rtime, res_fp, pend_samples[task_id])
        # while len(pend_samples) > 0:
        #     tasks = min(len(pend_samples), max_parallel)
        #     util.thread_it(eval_sim.eval_sample_task, [(manager, res_fp, pend_samples[row], row) for row in range(tasks)])
        #     _, pend_samples = archiver.import_dataset(res_fp, x_key="orig_input", y_key="target_err")


def build_calibration(manager, quiet=True):
    """Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
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

    DR_model = dim_red.create_DR(manager, quiet=quiet)

    if manager.add_nn_GP_mean:
        _, _, _, _, NN_mean_var = archiver.load_DR_settings(manager._settings_filepath)
        M_model = cgp.Mean_NN([manager.dim_in, manager.dim_out, *NN_mean_var], DR_model)
        M_model.calculate(manager, quiet=quiet)
        # auto saves to data folder
        model_fp = os.path.join(manager.model_dir, 'mean_model.pickle')
        archiver.save_model(M_model, model_fp)
    else:
        M_model = None
    return DR_model, M_model


def calibrate_simulation(manager, DR_model, M_model=None, max_parallel=2, quiet=True, use_emews=False):
    """Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
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
    for step in range(0, manager.num_BO_loops + 1):
        # After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
        eval_samples, pend_samples = manager.load_results()
        if abs(min(eval_samples[:, 0])) > manager.epsilon_stop:
            if use_emews:
                import proxies
                import eq
                import eval_wrapper

                func = proxies.dump_proxies(f=eval_wrapper.eval_dr_task)['f']
                # print(manager, DR_model, pend_samples)
                proxy_js = proxies.dump_proxies(manager=manager, dr_model=DR_model,
                                                pend_samples=pend_samples)
                exp_id = os.getenv("EXP_ID")
                payload = {'func': func, 'proxies': proxy_js, 'parameters': [{'row': r} for r in range(len(pend_samples))]}
                status, eq_task_id = eq.submit_task(exp_id, eq_type=0, payload=json.dumps(payload))
                if status != eq.ResultStatus.SUCCESS:
                    eq.stop_worker_pool(eq_type=0)
                    raise ValueError("Error submitting task while attempting to calibrate simulation")
                # timeout should be set to max duration of polaris run in seconds
                timeout = float(os.getenv("ME_TIMEOUT"))
                status, result = eq.query_result(eq_task_id, timeout=timeout)
                if status != eq.ResultStatus.SUCCESS:
                    eq.stop_worker_pool(eq_type=0)
                    raise ValueError("Error querying task result while attempting to calibrate simulation: {}".format(result))
                # args = [(manager, DR_model, pend_samples[row], row) for row in range(len(pend_samples))]
                # tmp_dir = os.path.join(os.environ.get("TURBINE_OUTPUT"), 'tmp')
                # pool = emews.Pool(tmp_dir, rank_type="workers")
                # pool.map(eval_sim.eval_DR_task, args)
            else:
                with futures.ThreadPoolExecutor(max_parallel) as executor:
                    result = executor.map(eval_sim.eval_DR_task, repeat(manager), repeat(DR_model), pend_samples, 
                                        (x for x in range(len(pend_samples))), repeat(False))
                    for obj, y_err, rtime, xhat, task_id in result:
                        eval_sim.update_DR_record(obj, y_err, rtime, pend_samples[task_id], xhat, manager)
                # util.thread_it(eval_sim.eval_DR_task, [(manager, DR_model, pend_samples[row], row) for row in range(len(pend_samples))])


        if DR_updates[0]:
            # TODO: this currently wipes out any pending recommended samples when updating
            if (step + 1) % DR_updates[1] == 0:
                DR_model = dim_red.tune_DR(manager, quiet=quiet)
                time.sleep(10)
        if manager.add_nn_GP_mean and mean_updates[0]:
            if (step + 1) % mean_updates[1] == 0:
                # need to re-arrange the results file as a new 'training file'
                M_model.tune(manager, quiet=quiet)
                model_fp = os.path.join(manager.model_dir, 'mean_model.pickle')
                archiver.save_model(M_model, model_fp)
                time.sleep(10)

        if step < manager.num_BO_loops:
            # If less then the number of trials we run, run another Bayes set
            print("running loop number {} of {}".format(step + 1, manager.num_BO_loops))
            bo.main_loop(manager, DR_model=DR_model, M_model=M_model)

    print("Review {} file in data directory".format(manager.res_filename))
