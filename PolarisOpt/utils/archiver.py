"""
    This file contains the master calls to handling the filing systems:
        * reading/writing in setting files
        * reading/writing in sample files
        * loading/saving models with dill
"""
import os
import numpy as np
import dill
import json
from . import util

def save_model(model, model_fp):
    r"""saves a model using dill

    Args:
        model: (model) the model to be saved. Example models include: DR_model, Mean_NN, GaussianProcess
        model_fp: (path) path to the location where the model should be saved
        
    Returns:
        prints a success indicator of model being saved and to where
    """
    with open(model_fp, "wb+") as f:
            dill.dump(model, f)
    return print('saved %s to %s' % (model, model_fp))


def load_model(model_fp):
    r"""loads a model using dill
    Args:
        model_fp: (path) path to the location where the model is saved
        
    Returns:
        model: (model) the loaded model. Example models include: DR_model, Mean_NN, GaussianProcess
    """
    if not os.path.exists(model_fp):
        raise ValueError("No saved model exists for %s" % model_fp)
    with open(model_fp, "rb") as f:
        model = dill.load(f)
    return model


def read_config(config_filepath):
    r"""Reads a json configuration file for the  calibration variables

    Args:
        config_filepath (path): the path location for the configuration file. See example_config.json for help
        
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
    variables = json.loads(open(config_filepath).read())   ##reads config file
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
        t_fp = os.path.join(dest_dir, vkey[0])
        records_list = json.loads(open(t_fp).read())
        for dkey in records_list:
            for ind in vkey[1]:
               if ind in records_list[dkey]:
                    records_list[dkey][ind], new_values = new_values[0], np.delete(new_values,0)
               else:
                   print(f'variable {ind} not found in {dkey}')
            
        with open(t_fp, 'w') as fp:
            json.dump(records_list, fp, indent = 4)


def load_DR_settings(DR_settings_filepath):
    r"""loads the parameters necessary for the applied Dimension Reduction technique

    Args:
        DR_settings_filepath: (path) path to the json file containing the Dimension Reduction controls
        
    Returns:
        method: the dimension reduction technique being applied
        dim_DR: the number of dimensions that the technique will produce the subspace over
        seed_value: the seed value for results consistancy
        NN_var: the necessary variables for a dimension-reduction neural network
        NN_mean_var: the necessary variables for a neural network mean for the Bayes Opt GP
    """
    records_list = json.loads(open(DR_settings_filepath).read())
    method = records_list['General DR controls']['method']
    dim_DR = records_list['General DR controls']['dim_DR']
    seed_value = records_list['General DR controls']['seed_value']
    NN_var = [records_list['DR neural network controls'][key] for key in records_list['DR neural network controls']]
        #[epochs, learning_rate, lambda, penalty, XDR_layer, DRX_layer, DRY_layer]
    NN_mean_var = [records_list['GP neural network mean controls'][key] for key in records_list['GP neural network mean controls']]
        #[epochs_m, learning_rate_m, layers_m]
    return method, dim_DR, seed_value, NN_var, NN_mean_var


def load_update_settings(settings_filepath):
    r"""loads the parameters necessary for the Bayes Opt to know if updating the subspace is needed

    Args:
        settings_filepath: (path) path to the json file containing the Dimension Reduction settings
        
    Returns:
        DR_updates: (list) contains a boolean True if the Dimension Reduction subspace should be retrained, followed 
                           by the interval for making the updates
        mean_updates: (list) contains a boolean True if the GP's NN-mean should be periodically retrained, followed 
                           by the interval for making the updates

    """
    records_list = json.loads(open(settings_filepath).read())
    DR_updates = [
        records_list['General DR controls']['method_update'], 
        records_list['General DR controls']['method_update_interval']]
    mean_updates = [
        records_list['General DR controls']['nn_mean_update'], 
        records_list['General DR controls']['nn_mean_update_interval']]
    return DR_updates, mean_updates


def import_dataset(training_filename, x_key = "orig_input", y_key = "target_err"):
    r"""A helper function to parse a file of datapoints in a json file
    
    Args:
        training_filename (filepath): the file containing the samples
        x_key (text): whether to return the inputs in the original domain ("orig_input") or the DR domain ("DR_input")
        y_key (text): whether to return the error by input ("target_err") or BO objective function ("objective")
    
    Return:
        eval_samples (nd-array): an array with each row documenting an evaluated sample
        pend_samples (nd-array): an array with each row documenting an unevaluated sample
    """
    try:
        records_list = json.loads(open(training_filename).read())
    except OSError:
        return np.array([]), np.array([])

    eval_samples = np.asarray([
        (item[y_key] +' ' + item[x_key]).split()
        for item in records_list if item["status"]=="Completed"
        ]).astype(float)        
    pends=[item[x_key] for item in records_list if item["status"]!="Completed"]
    if 'P' in pends:
        raise print("Some pending samples are untranslated in subspace %s and have been excluded" % x_key)
    else:
        pend_samples = np.asarray([
            item.split()
            for item in pends if item!='P'
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

def check_record_duplicate(x, records_list, identifier_key = "orig_input"):
    x =  util.convert_2str(x)
    for item in records_list:
        if item[identifier_key] == x:
            return True
    return False

def create_record(inputs, training_filename, var_names = None, identifier_key = "orig_input"):
    try:
        records_list = json.loads(open(training_filename).read())
    except OSError:
        records_list = []
    except json.decoder.JSONDecodeError:
        records_list = []
    for x in inputs:
        if check_record_duplicate(x, records_list, identifier_key):
            print(f'Duplicate Record will be skipped: {x}')
            continue
        else:
            new = new_record(x, var_names, identifier_key)
            records_list.append(new)
    with open(training_filename, 'w') as fp:
        json.dump(records_list, fp, indent = 4)


def update_record(inputs, keys, values, training_filename, identifier_key = "orig_input"):
    try:
        records_list = json.loads(open(training_filename).read())
    except OSError:
        records_list = []
        
    for x, v in zip(inputs, values):
        x = util.convert_2str(x)
        flag = 0
        for item in records_list:
            if item[identifier_key] == x:
                flag +=1
                for k, vv in zip(keys, v):
                    item[k] = util.convert_2str(vv)
        if flag == 0:
            new = new_record(x, identifier_key = identifier_key)
            for k, vv in zip(keys, v):
                new[k] = util.convert_2str(vv)
            records_list.append(new)
        if flag > 1:
            print("INFO: you have a repeated sample in %s" % training_filename)

    # print("Writing record to {}".format(training_filename), flush=True)
    with open(training_filename, 'w') as fp:
        json.dump(records_list, fp, indent = 4)