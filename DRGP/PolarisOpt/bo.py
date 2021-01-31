###NOTE: THIS DOES NOT CONSIDER SAMPLING AT ACKNOWLEDGED PENDING SAMPLES ALREADY IN RESULTS FILE###
#This version will use botorch with a matern ARD kernel. This acquisition function will be EI by default
#### ASSUMES THE SETUP OF THE FILE IS [Y, X]
import os
import sys
import numpy as np
import torch
import random
import gpytorch
from . import acquisition
from PolarisOpt import custom_gp as cgp
from .utils import sampler
from .utils import archiver
from .utils.fit import fit_gpytorch_torch

def main_loop(manager, DR_model = None, M_model = None):
    r"""main function to perform bayesian optimization
        Args:
            manager (SetupManager class): object containing paths to evaluated samples and settings for BO, specifically
                num_rec_points (int): number of points to recommend per iteration
                num_grid_points (int): number of potential candidates to be generated per iteration
                acq_type (string): the type of acquisition function to use, such as 'EI' or 'SPE'
                orig_range (nd-array): the original problem's variable ranges
            DR_model (class): The dimension reduction model if there is on
            M_model (class): The educated mean model if there is one

        Return:
            recommended samples are written to the results file with a 'P' designator for pending

        Example: 
            >>> if l<num_trials:
            >>>     BO.main_loop(manager)
    """
    if DR_model is not None:
        method = DR_model.method
        x_range = DR_model.DR_range
    else:
        method = 'None'
        x_range = manager.orig_range[0]
    
    eval_samples, pend_samples = manager.load_results()
    w = sys.stderr.write("#Evaluated Samples: %d #Pending: %d\n" % (np.shape(eval_samples)[0], np.shape(pend_samples)[0]))
    if len(eval_samples) == 0:
        raise ValueError('need to have a training set')
    elif manager.acq_type == 'EI':
        w = sys.stderr.write("Current Best: %f (sample %d)\n" % (np.min(eval_samples[:, 0]), np.argmin(eval_samples[:, 0])+1))

    unsam_pool = sampler.BO_pool(x_range, manager.num_grid_points, eval_samples, pend_samples) #in DR_subspace
    state_dict = None
    for r in range(manager.num_rec_points):
        gp = initialize_GP(eval_samples[:,1:], eval_samples[:,:1], M_model)
        best_x, unsam_pool = get_recommendation(gp, manager.acq_type, unsam_pool) #DR subspace
#        gp = gp.fantasize(best_x) #GP space
        archiver.create_record(best_x[:1,:], manager._res_filepath, manager.var, identifier_key = "DR_input")
        inputs = torch.as_tensor(best_x, device = gp.device) #in DR range
        posterior = gp.forward(inputs) #results in GP range
        fake_y = gp.likelihood(posterior).loc.detach()[:, None] #in GP range
        eval_samples=np.vstack((eval_samples,np.c_[fake_y.cpu().numpy(),best_x]))

def initialize_GP(train_X, train_Y, M_model = None, state_dict = None):
    r"""function to create a GP and optimize hyperparameters of 
    
    Args:
        train_X: 'n x d' array of training inputs
        train_Y: 'n x 1' array training outputs
        M_model (class): the educated mean model
        state_dict (dictionary): a starting point to improve processing

    Return:
        a optimized gaussian process 

    Example: 
        >>> train_X=np.random.randn(10, 2)
        >>> train_Y=np.random.randn(10, 1)
        >>> Mean_model = gpytorch.means.constant_mean.ConstantMean()
        >>> initialize_GP(train_X, train_Y, Mean_model)
    """
    gp = cgp.GaussianProcess(train_X, train_Y, mean_module = M_model)
    if state_dict is not None:
       gp.load_state_dict(state_dict)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.to(gp.device)
    mll.train()
    options = {}
    options["disp"] = False
    if gp.exclude_mean:
        options["exclude"] = [p_name for p_name, t in mll.named_parameters() if p_name.startswith('model.mean_module')] ##address this
    _ = fit_gpytorch_torch(mll, options=options)
    gp.eval()
    mll.eval()
    return gp


def get_recommendation(gp, acq_type, unsam_pool):
    r"""Choose next sample by determining acquisition function values, pick max
    and simulate a random 'observation' by generating a random value from the posterior of the selected point

        Args:
        gp (class): the pre-built gp
        acq_type (str): details which acuqisition function to apply
        unsam_pool (ndarray): list of potential candidates for recommendation in DR subspace
  
    Return:
        the recommended candidate and an updated sample pool

    Example: 
        >>> rec_x, new_pool = get_recommendation(gp, 'EI', unsam_pool)
    """
    acqf = acquisition.gen_acquisition(gp, acq_type) #in GP subspace
    acq_values, indices = acqf.forward(unsam_pool) #converts DR subspace to GP subspace within
    best_index = random.choice(indices) 
    sys.stderr.write("Pulled best from option of %s at %s\n" % (len(indices), acq_values[best_index].item()))
    best_x = unsam_pool[best_index:best_index+1] #in DR subspace
    sys.stderr.write("Selected sample %d from the unsampled pool.\n" % best_index)
    updated_unsam_pool = np.delete(unsam_pool, best_index, axis = 0) #in DR subspace
    return best_x, updated_unsam_pool #in DR subspace


