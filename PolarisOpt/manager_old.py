import os
import glob
import json
from .utils import archiver
from . import eval_sim

class Manager:
    r"""A helper class to house and manage all of the necessary files, parameters, and settings
        required to run package functions 
        
        Example:
        >>> settings_filepath = 'settings.json'
        >>> manager = Manager(settings_filepath)
        """
    def __init__(self, settings_filepath):
        r"""
        Args:
            settings_filepath (str): the filename for the json file containing the simulation, reductive subspace, 
                                  and Bayesian Optimization controls. This file must be located in the "data" folder.
                                  See "settings_readme.md" for more information
        Returns:
           a class object containing the information necessary to perform package functionality
           """
        self.epsilon_stop = 0.1
        self.num_BO_loops = 1
        self.num_rec_points = 1
        self.var = None
        self.vnames = None
        self.dim_in = None
        self.dim_out = None
        self.working_dir = os.getcwd() #temporary set to cwd
        self.settings_filepath =  self._check_file(settings_filepath) 
        self._load_jsonfile(self.settings_filepath)
        self._set_paths()
        self.run_id = 0
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        # print('validating SQL query on %s'%self.target_output_filepath)
        # self.dim_out = len(eval_sim.query_db(self.target_output_filepath, self.output_SQL_query))

    def get_dim_out(self):
        train, _ = archiver.import_dataset(self.training_filename, x_key = "orig_input", y_key = "target_err")
        self.dim_out = train.shape[1] - self.dim_in
        return self.dim_out
        # n =  len(eval_sim.query_db(self.target_output_filepath, self.output_SQL_query))
        # self.dim_out = n
        # return n
    
    def _check_file(self,file_path,create=False):
        if os.path.dirname(file_path)=='':
            file_path = os.path.join(self.working_dir, file_path)
        if not os.path.exists(file_path) and create:
            open(file_path,'w')    
        if not os.path.exists(file_path):
            raise ValueError('Path %s is invalid' % file_path)
        return file_path


    def _set_paths(self):
        self.training_filename = self._check_file(self.training_filename,True)
        self.res_filename      = self._check_file(self.res_filename,True)
        # self.target_output_filepath = self._check_file(self.target_output_filepath)
        self.model_dir = os.path.join(self.working_dir, 'Models')   #automatically saved in the results file folder
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # self.target_output_filename = os.path.basename(self.target_output_filepath)

    # Check with iterations finished which are not
    def check_iterations(self, task_dir,scenariopath):
        unfinishedit = []
        lastit = 0
        task_output = self.get_task_output(task_dir,scenariopath)
        for it in glob.glob(f'{task_output}_iteration_*'):
            if os.path.exists(os.path.join(it,'finished')):
                lastit = max(lastit,int(it.split('_')[-1]))
            else:
                unfinishedit.append(it)
        return lastit, unfinishedit
                
    def get_task_output(self, task_dir,scenariopath):
         d = json.loads(open(scenariopath).read())
         outpath = os.path.join(task_dir,d["Output controls"]['output_directory'])
         return outpath

    def _load_jsonfile(self, json_fp):
        if not json_fp.endswith('.json'):
            raise ValueError('File %s must be a json file' % json_fp)
        if not os.path.exists(json_fp):
            raise ValueError('File path %s is invalid' % json_fp)
        self.dictionary = json.loads(open(json_fp).read())
        # p = [vkey for vkey in self.dictionary]
        for vkey in ['General simulation controls', 'Convergence','File controls', 'General BO controls']:
            for key, value in self.dictionary[vkey].items(): 
                self.__dict__[key] = value 

    def update_parameter(self, name, value):
        if hasattr(self, name):
            self.__dict__[name] = value
            if "filename" in name:
                self._set_paths()
        else:
            raise ValueError("Parameter does not exist. If attempting to change a DR variable, this function is not needed; change in settings.json file directly")

    @property
    def res_model_filepath(self):
        n = os.path.split(self.res_filename.split(os.extsep)[0])[1]
        return os.path.join(self.model_dir,n + '_model.pickle')

    def load_training(self):
        if self.training_filename is None:
            raise ValueError('A training file path is required but has not been defined')
        if not os.path.exists(self.training_filename):
            raise ValueError('The current training data file path is invalid')
        train, _ = archiver.import_dataset(self.training_filename, x_key = "orig_input", y_key = "target_err")
        # if train.shape[1] != (self.dim_in + self.dim_out):
        #     raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
        return train[:, self.dim_out:], train[:, :self.dim_out]
        
    def load_results(self):
        if self._res_filepath is None:
            raise ValueError('A results file path is required but has not been defined')
        else:
            return archiver.import_dataset(self._res_filepath, x_key = "DR_input", y_key = "objective")

    def load_results_orig(self):
        if self._res_filepath is None:
            raise ValueError('A results file path is required but has not been defined')
        if not os.path.exists(self._res_filepath):
            raise ValueError('No original-subspace results file exists')
        train, _ = archiver.import_dataset(self._res_filepath, x_key = "orig_input", y_key = "objective")
        if train.shape[1] != (self.dim_in + 1):
            raise ValueError('Expected %s columns but got %s' % ((self.dim_in + 1), train.shape[1]))
        return train[:,1:], train[:,:1]

    def load_samples(self, filepath, x_key = "orig_input", y_key = "target_err"):
        if not os.path.exists(filepath):
            raise ValueError('%s is not a valid filepath' % filepath)
        archiver._check_keys(x_key,y_key) 
        eval_samples, uneval_samples = archiver.import_dataset(filepath, x_key, y_key)
        if y_key == "target_err":
            return eval_samples[:, self.dim_out:], eval_samples[:, :self.dim_out], uneval_samples
        else:
            return eval_samples[:,1:],eval_samples[:,:1], uneval_samples     





