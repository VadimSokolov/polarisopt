"""
    This file contains the master calls to implement:
        * Build_Sampleset - A latin hypercube sample generator and evaluator
        * Calibrate_Simulation - the dimension-reduced Bayesian Optimization calibration procedure 
"""

import os, sys
import numpy as np
import threading, shutil
import time, torch
from PolarisOpt import custom_gp as cgp
from . import sampler
from . import archiver
from PolarisOpt import eval_sim
from PolarisOpt import bo
from PolarisOpt import dim_red
import copy

def thread_it(f, a):
    r"""Generic Threading Function 
    Args:
        f (function): function to run in each thread
        a (list): list of argument instances within a () ex: [([10], 'save.txt'), ([11], 'save.txt')]
    Returns:
        joined threads
    """
    threads = []
    for item in a:
        t = threading.Thread(target = f, args = item)
        threads.append(t)
    
    for thread in threads:
        thread.start()
        time.sleep(20)
    
    for thread in threads:
        thread.join()
        #prevents us from doing anything until all are done



def build_sampleset(problem_info, res_filename, max_parallel = 2, num_samples = 0):
    r"""Function which runs all necessary steps to (create and) evaluate a sample training file.
    Args:
        problem_info (SetupManager class): central parameter keeper
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
        pend_samples = sampler.LHS_pool(problem_info.orig_range[0], num_samples, problem_info.orig_range[1])
        output = ''.join(["P " + ' '.join(map(str, i)) + "\n" for i in pend_samples])
        with open(res_filename,'a+') as outfile:
            outfile.write(output)
    else:
        _, pend_samples = archiver.import_dataset(res_filename)

    while len(pend_samples)>0:
        tasks = min(len(pend_samples), max_parallel)
        thread_it(eval_sim.eval_sample_task, [(problem_info, res_filename, pend_samples[row], row) for row in range(tasks)])
        _, pend_samples = archiver.import_dataset(res_filename)



def build_calibration(problem_info, pr = False):
    r"""Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
    Args:
        problem_info (SetupManager class): central parameter keeper
        pr (boolean): to (True) or not to (False) print training progress        
    Returns:
        DR_model (DR_Technique class): class containing dimension reduction technique
        M_model (Mean_NN class): class containing the GP NN mean if implemented
    """

    ########################
    # Setup and Train model#
    ########################

    DR_model = dim_red.create_DR(problem_info, pr)

    if problem_info.add_nn_GP_mean:
        _, _, _, _, NN_mean_var = archiver.load_DR_settings(problem_info.settings_filename)
        M_model = cgp.Mean_NN([problem_info.dim_in, problem_info.dim_out, *NN_mean_var], DR_model)
        M_model.calculate(problem_info, pr = pr)
        #auto saves to same folder as DR_model and res_filenames
        m_folder = os.path.join(os.path.dirname(problem_info.res_filename), 'Models')
        if not os.path.exists(m_folder):
            os.mkdir(m_folder)
        model_fn = os.path.join(m_folder, 'mean_model.pickle')
        archiver.save_model(M_model, model_fn)
    else:
        M_model = None
    return DR_model, M_model        



def calibrate_simulation(problem_info, DR_model, M_model = None, pr=False):
    r"""Function which runs all necessary steps to create models and run Bayesian Opt per settings.json file
    Args:
        problem_info (SetupManager class): central parameter keeper
        DR_model (DR_Technique class): class containing dimension reduction technique
        M_model (Mean_NN class): class containing the GP NN mean if implemented
        pr (boolean): to (True) or not to (False) print additional progress        
        
    Returns:
      a .pickle file containing the dimension-reduction model
      a .pickle file containing the mean model, if applicable
      a .dat file containing the Bayes Opt results in the objective function's single-dimension value and DR of format [Y, X]
      a .dat file containing the Bayes Opt results in the objective function's simulation-output dimensions of format [Y, X]
    """

    DR_updates, mean_updates = archiver.load_update_settings(problem_info.settings_filename)
        
    # going to have to run 1 more than number of requested loops to record the final loop's returned data points
    for l in range(0, problem_info.num_BO_loops+1):
        #After a Bayes set is recorded, we need to evaluate the pending ones denoted with 'P'
        eval_samples, pend_samples = problem_info.load_results()
        if abs(min(eval_samples[:,0])) > problem_info.epsilon_stop:
            thread_it(eval_sim.eval_DR_task, [(problem_info, DR_model, pend_samples[row], row) for row in range(len(pend_samples))])

        if DR_updates[0]:
            #TODO: this currently wipes out any pending recommended samples when updating
            if (l+1) % DR_updates[1] == 0:
                DR_model = dim_red.tune_DR(problem_info, pr)
                time.sleep(10)
        if problem_info.add_nn_GP_mean and mean_updates[0]:
            if (l+1) % mean_updates[1] == 0:
                #need to re-arrange the results file as a new 'training file'
                M_model.tune(problem_info, pr)
                m_folder = os.path.join(os.path.dirname(problem_info.res_filename), 'Models')
                model_fn = os.path.join(m_folder, 'mean_model.pickle')
                archiver.save_model(M_model, model_fn)
                time.sleep(10)

        if l<problem_info.num_BO_loops:
            #If less then the number of trials we run, run another Bayes set
            print("running loop number %d of %d" % (l+1, problem_info.num_BO_loops))
            bo.main_loop(problem_info, DR_model = DR_model, M_model = M_model)

    print("Review results.dat file in data directory")        

