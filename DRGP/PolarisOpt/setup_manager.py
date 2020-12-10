import os
import sys
import numpy as np
import json
from .utils import archiver
from . import eval_sim

class SetupManager:
    r"""A helper class to house and manage all of the necessary files, parameters, and settings
        required to run package functions 
        
        Example:
        >>> settings_filename = os.path.join(os.getcwd(), 'settings.json')
        >>> config_filename = os.path.join(os.getcwd(), 'config.json')
        >>> manager = SetupManager(settings_filename, config_filename)
        """
    def __init__(self, settings_filename, config_filename):
        r"""
        Args:
            settings_filename (path): the path to a json file containing the simulation, reductive subspace, 
                                  and Bayesian Optimization controls. See "settings_readme.md" 
                                  for more information
            config_filename (path):   the path to a json file containing information on the variables of the 
                                  simulation being optimized. See "example_config.json" for structure help

        Returns:
           a class object containing the information necessary to perform package functionality
           """
        self.working_dir = os.getcwd()
        self.epsilon_stop = 0.1
        self.num_BO_loops = 1
        self.num_rec_points = 1
        self.var = None
        self.vnames = None
        self.dim_in = None
        self.dim_out = None
        self.settings_filename = settings_filename
        self.config_filename = config_filename

    @property
    def settings_filename(self):
        return self._settings_filename

    @settings_filename.setter
    def settings_filename(self, file_fn):
        file_path = os.path.join(self.working_dir, 'data', file_fn)
        if not file_path.endswith('.json'):
            raise ValueError('The settings file must be a json file' % file_path)
        elif not os.path.exists(file_path):
            raise ValueError('The settings file path is invalid')
        self._settings_filename = file_path
        self._load_jsonfile(file_path)
        self._set_paths()        
        self.dim_out = len(eval_sim.query_db(self.target_output_filename, self.output_SQL_query))

    def _set_paths(self):
        data_path = os.path.join(self.working_dir,'data')
        self.training_filename = os.path.join(data_path, self.training_filename)
        self.res_filename = os.path.join(data_path, self.res_filename)
        self.target_output_filename = os.path.join(self.working_dir,'simulator','Target',self.target_output_filename)
       
    @property
    def config_filename(self):
        return self._config_filename

    @config_filename.setter
    def config_filename(self, file_fn):
        file_path = os.path.join(self.working_dir, 'data', file_fn)
        if not file_path.endswith('.json'):
            raise ValueError('The configuration file must be a json file')
        elif not os.path.exists(file_path):
            raise ValueError('The configuration file path is invalid')
        self._config_filename = file_path
        self.vnames, self.dim_in, self.orig_range = archiver.read_config(file_path) 
        self.var = [i for v in self.vnames for i in v[1]]

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
    def res_model_filename(self):
        d,n = os.path.split(self.res_filename.split(os.extsep)[0])
        return os.path.join(d, 'Models',n+'_model.pickle')

    def load_training(self):
        if self.training_filename is None:
            raise ValueError('A training file path is required but has not been defined')
        self.training_filename = os.path.join(self.working_dir,'data',self.training_filename)
        if not os.path.exists(self.training_filename):
            raise ValueError('The current training data file path is invalid')
        train, _ = archiver.import_dataset(self.training_filename, x_key = "orig_input", y_key = "target_err")
        if train.shape[1] != (self.dim_in + self.dim_out):
            raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
        return train[:, self.dim_out:], train[:, :self.dim_out]
        
    def load_results(self):
        if self.res_filename is None:
            raise ValueError('A results file path is required but has not been defined')
        else:
            return archiver.import_dataset(self.res_filename, x_key = "DR_input", y_key = "objective")

    def load_results_orig(self):
        if self.res_filename is None:
            raise ValueError('A results file path is required but has not been defined')
        if not os.path.exists(self.res_filename):
            raise ValueError('No original-subspace results file exists')
        train, _ = archiver.import_dataset(self.res_filename, x_key = "orig_input", y_key = "objective")
        if train.shape[1] != (self.dim_in + self.dim_out):
            raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
        return train[:, self.dim_out:], train[:, :self.dim_out]











