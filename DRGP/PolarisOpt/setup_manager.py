import os
import sys
import numpy as np
import json
from .utils import archiver

class SetupManager:
    r"""A helper class to house and manage all of the necessary files, parameters, and settings
        required to run package functions 
        
        Example:
        >>> settings_filename = os.path.join(os.getcwd(), 'settings.json')
        >>> config_filename = os.path.join(os.getcwd(), 'config.json')
        >>> problem_info = SetupManager(settings_filename, config_filename)
        """
    def __init__(self, settings_filename, config_filename):
        r"""
        Args:
            settings_filename (path): the path to a json file containing the simulation, reductive subspace, 
                                  and Bayesian Optimization controls. See "settings_readme.txt" 
                                  for more information
            config_filename (path):   the path to a json file containing information on the variables of the 
                                  simulation being optimized. See "example_config.json" for structure help

        Returns:
           a class object containing the information necessary to perform package functionality
           """
        self.settings_filename = settings_filename
        self.config_filename = config_filename
        self.working_dir = os.getcwd()
        self.epsilon_stop = 0.1
        self.num_BO_loops = 1
        self.num_rec_points = 1

    @property
    def settings_filename(self):
        return self._settings_filename

    @settings_filename.setter
    def settings_filename(self, file_path):
        if not file_path.endswith('.json'):
            raise ValueError('The settings file must be a json file' % file_path)
        elif not os.path.exists(file_path):
            raise ValueError('The settings file path is invalid')
        self._settings_filename = file_path
        self._load_jsonfile(file_path)

    @property
    def config_filename(self):
        return self._config_filename

    @config_filename.setter
    def config_filename(self, file_path):
        if not file_path.endswith('.json'):
            raise ValueError('The configuration file must be a json file')
        elif not os.path.exists(file_path):
            raise ValueError('The configuration file path is invalid')
        self._config_filename = file_path
        self.vnames, self.dim_in, self.orig_range = archiver.read_config(file_path) 

    def _load_jsonfile(self, json_fn):
            dictionary = json.loads(open(json_fn).read())
            p = [vkey for vkey in dictionary]
            for vkey in p[:3]:
                for key, value in dictionary[vkey].items(): 
                    self.__dict__[key] = value      

    def set_attribute(self, name, value):
        if hasattr(self, name):
            self.__dict__[name] = value
        else:
            raise ValueError("Attribute does not exist. If attempting to change a DR variable, change in settings.json file")

    @property
    def reso_filename(self):
        return self.res_filename.split(os.extsep)[0]+'_orig.'+self.res_filename.split(os.extsep)[1]

    @property
    def res_model_filename(self):
        return self.res_filename.split(os.extsep)[0]+'_model.pickle'

    def load_training(self):
        if not hasattr(self, 'dim_in') or not hasattr(self, 'dim_out'):
            raise ValueError('Please re-load the settings.json file')
        if self.training_filename is None:
            raise ValueError('A training file path is required but has not been defined')
        if not os.path.exists(self.training_filename):
            raise ValueError('The current training data file path is invalid')
        train, _ = archiver.import_dataset(self.training_filename)
        if train.shape[1] != (self.dim_in + self.dim_out):
            raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
        return train[:, self.dim_out:], train[:, :self.dim_out]
        
    def load_results(self):
        if self.res_filename is None:
            raise ValueError('A results file path is required but has not been defined')
        else:
            return archiver.import_dataset(self.res_filename)

    def load_results_orig(self):
        if not hasattr(self, 'dim_in') or not hasattr(self, 'dim_out'):
            raise ValueError('Please re-load the settings.json file')
        if self.reso_filename is None:
            raise ValueError('A results file path is required but has not been defined')
        if not os.path.exists(self.reso_filename):
            raise ValueError('No original-subspace results file exists')
        train, _ = archiver.import_dataset(self.reso_filename)
        if train.shape[1] != (self.dim_in + self.dim_out):
            raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
        return train[:, self.dim_out:], train[:, :self.dim_out]











