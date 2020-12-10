import os, sys
import numpy as np
import torch

def run_objective(Er, o_type='MSE'):
    if o_type == 'MSE':
        return Mean_Squared_Error(Er)
    else:
        raise ValueError('Undefined Objective Function')
    
##for when you have more than 1 y dim and need to create a single y dim
def Mean_Squared_Error(Er):
    r"""takes a sample and returns the mean squared error value
    """
    MSE=np.mean(Er**2,axis=-1)
    if len(MSE.shape)==1:
        MSE=MSE[:,None]
    return MSE,Er
