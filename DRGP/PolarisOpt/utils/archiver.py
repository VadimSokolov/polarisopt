"""
    This file contains the master calls to handling the filing systems:
        * reading/writing in setting files
        * reading/writing in sample files
        * loading/saving models with dill
"""
import os
import sys
import numpy as np
import dill
import json
from . import util

def save_model(model, model_fn):
    r"""saves a model using dill

    Args:
        model: (model) the model to be saved. Example models include: DR_model, Mean_NN, GaussianProcess
        model_fn: (path) path to the location where the model should be saved
        
    Returns:
        prints a success indicator of model being saved and to where
    """
    with open(model_fn, "wb+") as f:
            dill.dump(model, f)
    return print('saved %s to %s' % (model, model_fn))


def load_model(model_fn):
    r"""loads a model using dill
    Args:
        model_fn: (path) path to the location where the model is saved
        
    Returns:
        model: (model) the loaded model. Example models include: DR_model, Mean_NN, GaussianProcess
    """
    if not os.path.exists(model_fn):
        raise ValueError("No saved model exists for %s" % model_fn)
    with open(model_fn, "rb") as f:
        model = dill.load(f)
    return model


def read_config(config_filename):
    r"""Reads a json configuration file for the  calibration variables

    Args:
        config_filename (path): the path location for the 'config.json' file. See example_config.json for help
        
    Returns:
        variable names (nested list): an ordered list containing the names of every variable being explored, 
                                      partitioned by filename. Output will be of the form 
                                      [[file_name, [variable_1, variable_2...]], [filename_2, [variable_n, ...]]]
        number of dimensions (int): the number of variables being explored as outlined in the json file
        variable ranges (list): a matrix of the [min, max] bounds of each variable being explored in the original subspace 
                                      and a list of their corresponding types (float or int)

    Example:
        >>> vnames, dim_in, x_range = read_config(data_dir)
    """
    variables = json.loads(open(config_filename).read())   ##reads config file
#    totaldim = np.sum([vkey["size"] for key in variables for vkey in variables[key]])
    vnames = [[key, [vkey["name"] for vkey in variables[key] for n in range(0, int(vkey["size"]))]] for key in variables]
    Lower = [float(vkey["min"]) for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
    Upper = [float(vkey["max"]) for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
    Type = [vkey["type"] for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
    return vnames, len(Lower), [np.c_[Lower, Upper], Type]

def update_json(vnames, new_values, dest_dir):
    r"""Updates any json file(s) with new setting values

    Args:
        vnames: (list) a list of [file_name, [variable names]],...] to direct what to replace
        new_values: (n-array) of values to replace current variable assignment
        dest_dir: (path) directory holding the json file(s) to change
        
    Returns:
        updated json configuration file(s) in the named directory
    """
    #if we only have a single variable, convert scalar to array
    if np.isscalar(new_values):
        new_values = np.reshape(new_values, (1,1))[:, 0]
    elif len(new_values.shape) == 2:
        new_values = new_values[0,:]
    for vkey in vnames:
        t_fn = os.path.join(dest_dir, vkey[0])
        dictionary = json.loads(open(t_fn).read())
        for dkey in dictionary:
            for ind in vkey[1]:
                dictionary[dkey][ind], new_values = new_values[0], np.delete(new_values,0)
        with open(t_fn, 'w') as fp:
            json.dump(dictionary, fp, indent = 4)


def load_DR_settings(DR_settings_filename):
    r"""loads the parameters necessary for the applied Dimension Reduction technique

    Args:
        DR_settings_filename: (path) path to the json file containing the Dimension Reduction controls
        
    Returns:
        method: the dimension reduction technique being applied
        dim_DR: the number of dimensions that the technique will produce the subspace over
        seed_value: the seed value for results consistancy
        NN_var: the necessary variables for a dimension-reduction neural network
        NN_mean_var: the necessary variables for a neural network mean for the Bayes Opt GP
    """
    dictionary = json.loads(open(DR_settings_filename).read())
    method = dictionary['General DR controls']['method']
    dim_DR = dictionary['General DR controls']['dim_DR']
    seed_value = dictionary['General DR controls']['seed_value']
    NN_var = [dictionary['DR neural network controls'][key] for key in dictionary['DR neural network controls']]
        #[epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
    NN_mean_var = [dictionary['GP neural network mean controls'][key] for key in dictionary['GP neural network mean controls']]
        #[epochs_m, learning_rate_m, layers_m]
    return method, dim_DR, seed_value, NN_var, NN_mean_var


def load_update_settings(settings_filename):
    r"""loads the parameters necessary for the Bayes Opt to know if updating the subspace is needed

    Args:
        settings_filename: (path) path to the json file containing the Dimension Reduction settings
        
    Returns:
        DR_updates: (list) contains a boolean True if the Dimension Reduction subspace should be retrained, followed 
                           by the interval for making the updates
        mean_updates: (list) contains a boolean True if the GP's NN-mean should be periodically retrained, followed 
                           by the interval for making the updates

    """
    dictionary = json.loads(open(settings_filename).read())
    DR_updates = [
        dictionary['General DR controls']['method_update'], 
        dictionary['General DR controls']['method_update_interval']]
    mean_updates = [
        dictionary['General DR controls']['nn_mean_update'], 
        dictionary['General DR controls']['nn_mean_update_interval']]
    return DR_updates, mean_updates


def import_dataset(data_fn, x_key = "orig_input", y_key = "target_err"):
    r"""A helper function to parse a file of datapoints in a json file
    
    Args:
        data_fn (filepath): the file containing the samples
        x_key (text): whether to return the inputs in the original domain ("orig_input") or the DR domain ("DR_input")
        y_key (text): whether to return the error by input ("target_rr") or BO objective function ("objective")
    
    Return:
        eval_samples (nd-array): an array with each row documenting an evaluated sample
        pend_samples (nd-array): an array with each row documenting an unevaluated sample
    """
    try:
        dictionary = json.loads(open(data_fn).read())
    except OSError:
        return np.array([]), np.array([])

    eval_samples = np.asarray([
        (item[y_key] +' ' + item[x_key]).split()
        for item in dictionary if item["status"]=="Completed"
        ]).astype(float)        
    pend_samples = np.asarray([
        item[x_key].split()
        for item in dictionary if item["status"]!="Completed"
        ]).astype(float)        
    return eval_samples, pend_samples

def new_record(inputs, var_names = None, identifier_key = "orig_input"):
    i = util.convert_2str(inputs)
    if var_names is None:
        v = "P"
    else:
        v = util.convert_2str(var_names)
    new = {
        "status": "Pending",
        "variables": v,
        }
    if identifier_key == "DR_input":
        new.update({
            "orig_input": "P",
            "DR_input": i,
            })
    else:
        new.update({
            "orig_input": i,
            "DR_input": "P"
            })
    new.update({
        "target_err": "P",
        "objective": "P",
        "run_time": "P"
        })
    return new   

@ -208,16 +208,13 @@ def create_record(inputs, data_fn, var_names = None, identifier_key = "orig_input"):
    try:
        dictionary = json.loads(open(data_fn).read())
    except OSError:
        dictionary = []

    for item in inputs:
        new = new_record(item, var_names, identifier_key)
        dictionary.append(new)

    with open(data_fn, 'w') as fp:
        json.dump(dictionary, fp, indent = 4)


def update_record(inputs, keys, values, data_fn, identifier_key = "orig_input"):
    try:
        dictionary = json.loads(open(data_fn).read())
    except OSError:
        dictionary = []
    try:
        v_list= list(zip(*values))
    except:
        v_list= values
        
    for i, v in zip(inputs, v_list):
        ii = util.convert_2str(i)
        flag = 0
        for item in dictionary:
            if item[identifier_key] == ii:
                flag +=1
                for k, vv in zip(keys, v):
                    item[k] = util.convert_2str(vv)
        if flag == 0:
            new = new_record(ii, identifier_key = identifier_key)
            for k, vv in zip(keys, v):
                new[k] = util.convert_2str(vv)
            dictionary.append(new)
        if flag > 1:
            print("INFO: you have a repeated sample in %s" % data_fn)

    with open(data_fn, 'w') as fp:
        json.dump(dictionary, fp, indent = 4)
