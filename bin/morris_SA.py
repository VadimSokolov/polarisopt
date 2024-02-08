from SALib.sample import morris as morris_s
from SALib.sample import latin as latin_s
from SALib.analyze import morris as morris_a
import SALib
import numpy as np
import os
import sys
# import PolarisOpt
from PolarisOpt.utils import archiver
from PolarisOpt.F import build_sampleset
from PolarisOpt.setup_manager import SetupManager

#######################################
#######################################
# Pull in all of the settings for use #
#######################################
#######################################


settings_filepath = 'scratch/settings_slurm.json'
config_filepath = 'scratch/config_morris_timing.json'
manager=SetupManager(settings_filepath, config_filepath)

problem = {
    'num_vars': manager.dim_in,
    'names': manager.var,
    'bounds': manager.orig_range[0]
}

# n = 4
# test_problem = {'num_vars':n,'names': range(n),'bounds': [[0,1]]*n}
# morris_s.sample(test_problem,N=25,num_levels=4).shape

# manager.get_dim_out()
import pickle
pickle.dump(problem, open('scratch/problem.pkl','wb'))

manager.run_id = 0
if os.stat(manager.training_filename).st_size == 0:
    print("Creating training set")
    X = morris_s.sample(problem,N=8,num_levels=4)
    # X = latin_s.sample(problem,N=40)
    print(f'X shape: {X.shape}')
    archiver.create_record(X, manager.training_filename, var_names = manager.var, identifier_key = "orig_input")

# np.unique(X, axis=0, return_counts=True)
print(f'Using Slurm: {manager.dictionary["slurm"]["useslurm"]}')
print("Starting Evaluation of Sample")
build_sampleset(manager, manager.training_filename, max_parallel=40,num_samples=0)


# We do the analysis in a separate file now: morris-eda.ipynb
# X,Err = manager.load_training()
# Obj, _ = objective_funcs.run_objective(Err, o_type=manager.objective_type)


# Si = morris_a.analyze(problem, X, Obj, conf_level=0.95, num_levels=4)