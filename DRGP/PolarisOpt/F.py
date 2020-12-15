"""
    This file contains the main function calls
"""

import os, sys
import numpy as np
import time
from PolarisOpt import custom_gp as cgp
from PolarisOpt.utils import sampler
from PolarisOpt.utils import archiver
from PolarisOpt.utils import util
from PolarisOpt import eval_sim
from PolarisOpt import bo
from PolarisOpt import dim_red


def build_sampleset(manager, res_filename, max_parallel = 2, num_samples = 0, use_emews=False):
    r"""Function which runs all necessary steps to (create and) evaluate a sample training file.
    Args:
        manager (SetupManager class): central parameter keeper
        res_filename (path): the file to place evaluated or pending points into. Of the format [Y,X]
        max_parellel (int): the largest number of parallel evaluations allowed while evaluating all pending samples
                in the res_filename file
        num_samples (int): the number of samples taken from a Lating Hypercube constructed across the statespace
                If num_samples = 0, no additional samples will be created
        
    Returns:
      a file containing the evaluated samples in the format necessary for training [Y, X]
    """

    #################################
    #STEP 1: Create LHS if desired  #
    #################################
    if num_samples>0:
        pend_samples = sampler.LHS_pool(manager.orig_range[0], num_samples, manager.orig_range[1])
        archiver.create_record(pend_samples, res_filename, var_names = manager.var, identifier_key = "orig_input")
    else:
        _, pend_samples = archiver.import_dataset(res_filename, x_key = "orig_input", y_key = "target_err")

    if use_emews:
        import emews
        args = [(manager, res_filename, pend_samples[row], row) for row in range(len(pend_samples))]
        tmp_dir = os.path.join(os.environ.get("TURBINE_OUTPUT"), 'tmp')
        pool = emews.Pool(tmp_dir, rank_type="workers")
        pool.map(eval_sim.eval_sample_task, args)
    else:
        while len(pend_samples)>0:
            tasks = min(len(pend_samples), max_parallel)
            util.thread_it(eval_sim.eval_sample_task, [(manager, res_filename, pend_samples[row], row) for row in range(tasks)])
            _, pend_samples = archiver.import_dataset(res_filename, x_key = "orig_input", y_key = "target_err")



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
        _, _, _, _, NN_mean_var = archiver.load_DR_settings(manager.settings_filename)
        M_model = cgp.Mean_NN([manager.dim_in, manager.dim_out, *NN_mean_var], DR_model)
        M_model.calculate(manager, quiet = quiet)
        #auto saves to same folder as DR_model and res_filenames
        m_folder = os.path.join(os.path.dirname(manager.res_filename), 'Models')
        if not os.path.exists(m_folder):
            os.mkdir(m_folder)
        model_fn = os.path.join(m_folder, 'mean_model.pickle')
        archiver.save_model(M_model, model_fn)
    else:
        M_model = None
    return DR_model, M_model        



def calibrate_simulation(manager, DR_model, M_model = None, quiet=True, use_emews=False):
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

    DR_updates, mean_updates = archiver.load_update_settings(manager.settings_filename)
        
    # going to have to run 1 more than number of requested loops to record the final loop's returned data points
    for l in range(0, manager.num_BO_loops+1):
        #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
        eval_samples, pend_samples = manager.load_results()
        if abs(min(eval_samples[:,0])) > manager.epsilon_stop:
            if use_emews:
                import emews
                args = [(manager, DR_model, pend_samples[row], row) for row in range(len(pend_samples))]
                tmp_dir = os.path.join(os.environ.get("TURBINE_OUTPUT"), 'tmp')
                pool = emews.Pool(tmp_dir, rank_type="workers")
                pool.map(eval_sim.eval_DR_task, args)
            else:
               util.thread_it(eval_sim.eval_DR_task, [(manager, DR_model, pend_samples[row], row) for row in range(len(pend_samples))])

        if DR_updates[0]:
            #TODO: this currently wipes out any pending recommended samples when updating
            if (l+1) % DR_updates[1] == 0:
                DR_model = dim_red.tune_DR(manager, quiet = quiet)
                time.sleep(10)
        if manager.add_nn_GP_mean and mean_updates[0]:
            if (l+1) % mean_updates[1] == 0:
                #need to re-arrange the results file as a new 'training file'
                M_model.tune(manager, quiet = quiet)
                m_folder = os.path.join(os.path.dirname(manager.res_filename), 'Models')
                model_fn = os.path.join(m_folder, 'mean_model.pickle')
                archiver.save_model(M_model, model_fn)
                time.sleep(10)

        if l<manager.num_BO_loops:
            #If less then the number of trials we run, run another Bayes set
            print("running loop number %d of %d" % (l+1, manager.num_BO_loops))
            bo.main_loop(manager, DR_model = DR_model, M_model = M_model)

    print("Review %s file in data directory"% manager.res_filename)        

