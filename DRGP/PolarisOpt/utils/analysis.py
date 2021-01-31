
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from matplotlib import cm


def track_sol(yhat,minimization=True):
    r"""A helper function to parse the results file for the final recommended solution and sampling history
    
    Args:
        yhat (ndarray): an array of the objective values pulled from the results file
        minimization (Boolean): an indicator if the tracking should be over the minimum or maximum recommendation
    
    Return:
        best value (float): best max/min found among all samples
        index (list): the sample number(s) for the instance(s) of the best value found
        max/min history (ndarray): the improvement history as samples are evaluated
    """
    if not minimization:
        yhat=-yhat
    mv=np.array(yhat[0:1])
    indx=[]
    for i in range(0,len(yhat)):
        if yhat[i]<=mv[-1]:
            mv=np.append(mv,yhat[i])
        else:
            mv=np.append(mv,mv[-1])
    indx=np.where(yhat==mv[-1])
    if not minimization:
        mv=-mv
    return mv[-1],indx,mv


#[fn1,fn2,fn3...]
def analyze_min(fn_list,minimization=True):
    r"""A helper function to parse a set of files
    
    Args:
        fn_list (list): an array of the objective values pulled from the results file
        minimization (Boolean): an indicator if the tracking should be over the minimum or maximum recommendation
    
    Return:
        best value (float): best max/min found among all samples
        index (list): the sample number(s) for the instance(s) of the best value found
        max/min history (ndarray): the improvement history as samples are evaluated
    """
    yhats=[]
    for filename in fn_list:
        if os.path.exists(filename):
            #in case there are some unevaluated samples
            all_samples=np.loadtxt(filename,dtype=np.str,delimiter=" ")
            eval_samples=all_samples[all_samples[:,0]!="P",:].astype(float)
        else:
            raise ValueError(filename+' not found')
        #find mins
        yhats=[*yhats,minimization_tracking(eval_samples[:,0],minimization)]
    return yhats

