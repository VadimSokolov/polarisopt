import os, glob,shutil
import json
from concurrent import futures
from PolarisOpt.slurm_wrappers import run_sim_slurm

class Manager:
    r"""A helper class to house and manage all of the necessary files, parameters, and settings
        required to run package functions 
        """
    def __init__(self, settings_filepath, s):
        r"""
        Args:
            settings_filepath (str): the filename for the json file containing the simulation, reductive subspace, 
                                  and Bayesian Optimization controls. This file must be located in the "data" folder.
                                  See "settings_readme.md" for more information
            s : Sampler
                Sampler object that inplements getsample() function to be used to get the sample object
        Returns:
           a class object containing the information necessary to perform package functionality
           """
        self.settings_filepath =  self._check_file(settings_filepath) 
        self._settingsfromjson()
        if not hasattr(self,'working_dir'): #in case working_dir is not defined in settings.json
            self.working_dir = os.getcwd() 
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        self.s = s #sampler object
 
    def _check_file(self,file_path,create=False):
        if os.path.dirname(file_path)=='':
            file_path = os.path.join(self.working_dir, file_path)
        if not os.path.exists(file_path) and create:
            fh = open(file_path,'w'); fh.close()
        if not os.path.exists(file_path):
            raise ValueError('Path %s is invalid' % file_path)
        return file_path
    def _settingsfromjson(self):
        json_fp = self.settings_filepath
        self.dictionary = json.loads(open(json_fp).read())
        # p = [vkey for vkey in self.dictionary]
        # for vkey in ['General simulation controls', 'Convergence','File controls', 'General BO controls']:
        for vkey in ['General simulation controls', 'Convergence','File controls']:
            for key, value in self.dictionary[vkey].items(): 
                self.__dict__[key] = value 
    def run_study(self):
        next_sample = self.s.getsamples(max = self.num_parallel_runs)
        # for x in next_sample:
        #     self.run_task(x)

        while len(next_sample)>0:
            with futures.ThreadPoolExecutor(self.num_parallel_runs) as executor:
                result = executor.map(self.run_task, next_sample)
            for r in result:
                ex = getattr(r, "exception", None)
                if ex is None:
                    pass # No exception was raises
                else:
                    print(r.excepition())
            next_sample = self.s.getsamples(max = self.num_parallel_runs)
    
    def run_task(self, sample):
        completeflag, last_it = self.check_sample_complition(sample.folder)
        if completeflag:
            sample.status = "Completed"
            print(f"Sample in folder {sample.folder} was already complete. Not adding this task to the queue.")
            return sample
        folder = os.path.join(self.working_dir, f'experiments/Sim{str(sample.index)}')
        if not completeflag and os.path.exists(folder):
            print(f'INFO: folder: {folder} was created but the task was not completed, removing the outputs')
            outf = self.get_task_output(folder)
            l = glob.glob(f'{outf}*')
            for dir in filter(os.path.isdir, l):
                shutil.rmtree(dir)
        
        self.create_simulation_folder(folder)
        sample.folder = folder
        sample.start_iteration_from = 1
        for i in range(self.s.p):
            self.update_json(self.s.varnames[i],sample.input[i],os.path.join(sample.folder,self.s.varfiles[i]))
        if self.dictionary["slurm"]["useslurm"]:
            sample = run_sim_slurm(sample,self)
        else:
            scenariopath = os.path.join(sample.folder, self.polaris_scenario_file)
            print('Running the simulation locally', flush=True)
            res = run_sim_local(sample.folder, self.polaris_executable, scenariopath, self.working_dir, convrgencepath)
        if res is True and self.check_sample_complition(sample.folder):
            sample.status = 'finished'
        else:
            sample.status = 'failed'
        return sample    

    def check_sample_complition(self, task_dir):
        # Returns is_complete flag and the number of last completed iterations if convergece is used
        res = (False, None)
        if task_dir is None:
            return res
        # check if task was already executed
        if self.convergence:
            lastit, _ = self.check_iterations(task_dir)
            if lastit >= self.num_abm_runs:
                # print(f"Simulation {task_dir} was already run {self.num_abm_runs} times. Not adding this task to the queue.")
                res = (True, lastit)
            else:
                res = (False, lastit)
        else:
            if os.path.exists(task_dir):
                task_output = self.get_task_output(task_dir)
                if os.path.exists(os.path.join(task_output,'finished')):
                    # print(f"Finished file was created in {task_output}. Not adding this task to the queue.")
                    res = (True,None)
        return res
                
    
    def create_simulation_folder(self, task_dir):
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)    
            print(f'Copying simulation files to {task_dir}', flush=True)
            self.copy_simulation(self.simulation_path, task_dir)
    def copy_simulation(self,src_dir, dst_dir):
        names = os.listdir(src_dir)
        for name in names:
            src_name = os.path.join(src_dir, name)
            if not os.path.isdir(src_name):
                dst_name = os.path.join(dst_dir, name)
                shutil.copy2(src_name, dst_name)
    def update_json(self,vname, new_value, file):
        r"""Updates any json file(s) with new setting values. It assumed that the file has a two-layer structure. First dictionary contains categories ov variables. Second layer (nested) dictionary contains the actual variables. 

        Args:
            vname: (str) name of the variable to be replaced
            new_value: value to replace current variable assignment
            file: (path) full path for the json file holding variable to be  changed
            
        Returns:
            updated json configuration file(s) in the named directory
        """
        if not os.path.exists(file):
            print(f'INFO: update_json: file {file} does not exist, not doing anything')
            return
        with open(file,'r') as fp:
            records_list = json.loads(fp.read())
        flag = False
        for dkey in records_list:
            if vname in records_list[dkey]:
                records_list[dkey][vname] = new_value
                flag = True      
        if not flag: # just print a warning
            print(f'WARNING: Variable {vname} was not found in file {file}')
        else: # update the file 
            with open(file, 'w') as fp:
                json.dump(records_list, fp, indent = 4)

    # # Check with iterations finished which are not
    def check_iterations(self, task_dir):
        unfinishedit = []
        lastit = 0
        task_output = self.get_task_output(task_dir)
        for it in glob.glob(f'{task_output}_iteration_*'):
            if os.path.exists(os.path.join(it,'finished')):
                lastit = max(lastit,int(it.split('_')[-1]))
            else:
                unfinishedit.append(it)
        return lastit, unfinishedit
                
    def get_task_output(self, task_dir):
        scenariopath = os.path.join(task_dir, self.polaris_scenario_file)
        with open(scenariopath) as fh:
            d = json.loads(fh.read())
            outpath = os.path.join(task_dir,d["Output controls"]['output_directory'])
            return outpath





    # def update_parameter(self, name, value):
    #     if hasattr(self, name):
    #         self.__dict__[name] = value
    #         if "filename" in name:
    #             self._set_paths()
    #     else:
    #         raise ValueError("Parameter does not exist. If attempting to change a DR variable, this function is not needed; change in settings.json file directly")

    # @property
    # def res_model_filepath(self):
    #     n = os.path.split(self.res_filename.split(os.extsep)[0])[1]
    #     return os.path.join(self.model_dir,n + '_model.pickle')

    # def load_training(self):
    #     if self.training_filename is None:
    #         raise ValueError('A training file path is required but has not been defined')
    #     if not os.path.exists(self.training_filename):
    #         raise ValueError('The current training data file path is invalid')
    #     train, _ = archiver.import_dataset(self.training_filename, x_key = "orig_input", y_key = "target_err")
    #     # if train.shape[1] != (self.dim_in + self.dim_out):
    #     #     raise ValueError('Expected %s columns but got %s' % ((self.dim_in + self.dim_out), train.shape[1]))
    #     return train[:, self.dim_out:], train[:, :self.dim_out]
        
    # def load_results(self):
    #     if self._res_filepath is None:
    #         raise ValueError('A results file path is required but has not been defined')
    #     else:
    #         return archiver.import_dataset(self._res_filepath, x_key = "DR_input", y_key = "objective")

    # def load_results_orig(self):
    #     if self._res_filepath is None:
    #         raise ValueError('A results file path is required but has not been defined')
    #     if not os.path.exists(self._res_filepath):
    #         raise ValueError('No original-subspace results file exists')
    #     train, _ = archiver.import_dataset(self._res_filepath, x_key = "orig_input", y_key = "objective")
    #     if train.shape[1] != (self.dim_in + 1):
    #         raise ValueError('Expected %s columns but got %s' % ((self.dim_in + 1), train.shape[1]))
    #     return train[:,1:], train[:,:1]

    # def load_samples(self, filepath, x_key = "orig_input", y_key = "target_err"):
    #     if not os.path.exists(filepath):
    #         raise ValueError('%s is not a valid filepath' % filepath)
    #     archiver._check_keys(x_key,y_key) 
    #     eval_samples, uneval_samples = archiver.import_dataset(filepath, x_key, y_key)
    #     if y_key == "target_err":
    #         return eval_samples[:, self.dim_out:], eval_samples[:, :self.dim_out], uneval_samples
    #     else:
    #         return eval_samples[:,1:],eval_samples[:,:1], uneval_samples     





