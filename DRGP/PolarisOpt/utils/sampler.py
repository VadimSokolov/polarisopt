import os
import numpy as np
from pyDOE import lhs

def LHS_pool(x_range, grid_num, x_type = []):
    r"""Produces a list of potential candidates via a latin hypercube across the entire domain 
    Args:
        x_range (ndarray): the ranges for each of the dimensions [xlb, xub]
        grid num (int): the number of samples that should be produced
        x_type (list): the variable type for each variable with; only 'int' types are manipulated
    
    Returns: 
        sample pool (ndarray): array of potential samples across the entire subspace
    """
    norm_sam = lhs(x_range.shape[0], samples = grid_num)
    dn_sam = norm_sam*(x_range[:, 1]- x_range[:, 0]) + x_range[:, 0]
    #adjust if our original variables were integers
    if not x_type:
        return dn_sam
    else:
        for i in range(0, len(x_range)):
            if x_type[i] == 'int':
                dn_sam[:, i] = np.round(dn_sam[:,i]).astype('int')
    return dn_sam

def BO_pool(x_range, grid_num, eval_samples, pend_samples):
    r"""A helper function to produce the list of BO potential candidates by
        creating a latin hypercube across the entire domain and a second
        hypercube centered around the current minimum sample
    Args:
        x_range (ndarray): the ranges for each of the dimensions used in the GP [xlb, xub]
        grid_num (int): the number of samples that should be produced
        eval_samples (ndarray): an array with each row documenting evaluated sample
        pend_samples (ndarray): an array with each row documenting unevaluated samples
    
    Returns: 
        sample pool (ndarray): rows of samples for potential selection by the BO
    """
    #TO DO: Update this and GP to have an 'int','float' designator list depending on GP transformations
    #run the latin hypercube for the entire domain
    pool = LHS_pool(x_range, grid_num)
	#encourage exploitation with a scatter of points around the current best
    dim_min = x_range[:, 0]
    dim_max = x_range[:, 1]
    sample_pool = np.vstack((dim_min+((dim_max-dim_min)/2), dim_min, dim_max, pool))
    if len(eval_samples)>0:
        best = np.argmin(eval_samples[:, 0])
        scatter = LHS_pool(np.c_[-.00005*np.ones(len(x_range)), .00005*np.ones(len(x_range))], 50)+eval_samples[best, 1:]
        sample_pool = np.unique(np.vstack((scatter, sample_pool)), axis = 0)
    return enforce_unique(sample_pool,eval_samples,pend_samples)


def enforce_unique(sample_pool, eval_samples = [], pend_samples = []):
    r"""A helper function to ensure that a sample pool is made of only unconsidered samples 
    Args:
        sample_pool (ndarray): the sample set being adjusted
        eval_samples (ndarray): an array with each row documenting previously evaluated sample
        pend_samples (ndarray): an array with each row documenting previously recommended but unevaluated samples
    Returns: 
        sample pool (ndarray): rows of untried samples for potential selection by the BO
    """
    dim = sample_pool.shape[1]
    if eval_samples != []:
        if eval_samples.shape[1]-1 != dim:
            raise ValueError("size mismatch, eval_samples: [%s x %s], sample_pool: [%s x %s]" % (*eval_samples.shape, *sample_pool.shape))
        else:
            for item in eval_samples:
                sample_pool = np.delete(sample_pool, np.where(np.all(sample_pool == item[1:], axis = 1)), axis = 0)
    if pend_samples != []: 
        if pend_samples.shape[1] != dim:
            raise ValueError("size mismatch, pend_samples: [%s x %s], sample_pool: [%s x %s]" % (*pend_samples.shape, *sample_pool.shape))
        else:
            for item in pend_samples:
                sample_pool = np.delete(sample_pool, np.where(np.all(sample_pool == item, axis = 1)), axis = 0)
    return sample_pool