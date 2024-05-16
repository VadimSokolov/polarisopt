from polarislib.runs.convergence.convergence_config import ConvergenceConfig
from polarislib.runs.convergence.convergence_runner import run_polaris_convergence

project_dir=$PRJDIR
polaris_binary=$POLARISBIN
config = ConvergenceConfig(data_dir     = project_dir, 
                           do_abm_init  = False,
                           do_skim      = False,
                           db_name      = $DBNAME, 
                           num_abm_runs = $NRUNS, 
                           num_threads  = $NCPUS,
                           num_retries = 3,
                           start_iteration_from = $RESTART,
                           polaris_exe = polaris_binary)
run_polaris_convergence(config)