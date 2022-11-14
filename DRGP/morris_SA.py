from SALib.sample import morris as morris_s
from SALib.analyze import morris as morris_a
import numpy as np
import os
import sys
import PolarisOpt
from PolarisOpt.utils import archiver
from PolarisOpt.utils import objective_funcs

#######################################
#######################################
# Pull in all of the settings for use #
#######################################
#######################################

settings_filename = 'settings.json'
config_filename = 'config_morris.json'
manager=PolarisOpt.setup_manager.SetupManager(settings_filename, config_filename)


problem = {
    'num_vars': manager.dim_in,
    'names': manager.var,
    'bounds': manager.orig_range[0]
}


X = morris_s.sample(problem,N=2,num_levels=4)
archiver.create_record(X, manager._training_filepath, var_names = manager.var, identifier_key = "orig_input")
PolarisOpt.F.build_sampleset(manager, manager._training_filepath, max_parallel=32,num_samples=0)

X,Err = manager.load_training()
Obj, _ = objective_funcs.run_objective(Err, o_type=manager.objective_type)


Si = morris_a.analyze(problem, X, Obj, conf_level=0.95, num_levels=4)