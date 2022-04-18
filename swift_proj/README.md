# EMEWS Calibration Workflow

Running Locally:

1. Start the DB
2. Edit data/cfgs/local_worker_pool.cfg to update the site file, if necessary
3. ./local_worker_pool.sh exp_id ../data/cfgs/local_worker_pool.cfg
4. ./run_drgp.sh exp_id ../data/cfgs/drgp.cfg

Running on Bebop

1. Start the DB
    * source db/env-bebop.sh
    * db/db-start.sh (note the DB_HOST)
2. source `envs/bebop_env.sh`
3. Edit data/cfgs/bebop_worker_pool.cfg to update the CFG_DB_HOST if necessary.
4. Edit data/cfgs/bebop_drgp.cfg to update the CFG_DB_HOST, and CFG_DRGP_TIMEOUT if necessary
5. swift/bebop_worker_pool.sh exp_id data/cfgs/bebop_worker_pool.cfg, and note job id.
6. scripts/bebop_submit_drgp.sh exp_id data/cfgs/bebop_drgp.cfg pool_job_id. exp_ids should be the same for both.




## OLD EQ/PY - PYMAP WORKFLOW INSTRUCTIONS

The EMEWS calibration workflow runs the PolarisOpt calibration using EMEWS to dispatch
the polaris runs. The python code (i.e., `eval_sim.eval_DR_task` and `eval_sim.eval_sample_task`)
that launches each polaris run is run in its own Python interpreter. 

## Dependencies

The calibration workflow requires additional python packages to be installed.

```
# load the correct Python
module load anaconda3/2020.07
# install pytorch
pip3 install torch --user
# install gpytorch
pip3 install gpytorch --user
# install botorch v.0.1.4
pip3 install botorch==0.1.4 --user
pip3 install dill --user
pip3 install pyDOE --user
```



## Running the Workflow on Bebop

```
# cd to swift_proj
cd swift_proj
# setup the environment for Bebop
source envs/bebop_env.sh
# cd to swift
cd swift
# submit the workflow to the slurm scheduler
./bebop_run_drgp.sh <experiment_id> <cfg_file>
```

The arguments to the `bebop_run_drgp.sh` submit script are:

* experiment_id - an experiment identifier id
* cfg_file - a workflow configuration file

For example,

```
bebop_run_drgp.sh my_calibration_1 cfgs/my_bebop_calibration.cfg
```

The working directory for the calibration will be `swift_proj/experiments/<experiment_id>`
(the experiment directory) where experiment_id is the *<experiment_id>* command line argument. 

Within that directory, the workflow creates the following files:

* output.txt - anything written to standard out is written to this file
* experiments/SimN - the working directory for an individual polaris run (*Sim0*, *Sim1*, etc.)
* pymap/out_N.txt - the output from the polaris run *SimN* and the python code that launched it
* pymap/err_N.txt - the error output the polaris run *SimN* and the python code that launched it

In addition the workflow algorithm json file (see below) and the calibration settings and config json files
are copied to the experiment directory. These copied files are used during the calibration
and the calibration results will be written to them, _NOT_ to the original files.

## Configuration

There are two configuration files:

* job configuration file (e.g., `cfgs/bebop.cfg`) that is passed to the job submission script
* workflow algorithm configuration (e.g., `swift_proj/data/algo_params/test_params.json`) that
is passed to the workflow algorithm (`swift_proj/python/drgp.py`).

### Job Configuration

The job configuration file is a bash script that is sourced by the submission script (i.e., 
`bebop_run_drgp.sh`) to set the configuration (walltime, etc.) for the job, as well as
the location of the workflow algorithm configuration file. See `swift_proj/swift/cfgs/bebop_template.cfg` for an example. That file can be copied and edited for a particular workflow run.

The job configuration file should have the following entries:

* CFG_WALLTIME - jobs walltime
* CFG_PROCS - the requested number of processes (tasks). *Note* that EMEWS requires 2 processes for itself,
so the number of available workers with CFG_PROCS - 2.
* CFG_PPN - the number of processes per node. On Bebop, the number of threads passed to polaris is
36 / CFG_PPN.
* CFG_QUEUE - the queue to use (e.g., bdwall on bebop).
* CFG_PROJECT - the project to use for the job
* CFG_ALGO_PARAMS - the path the workflow algorithm configuration file
* CFG_SITE_FILE - the site configuration file. On Bebop, this should be `swift_proj/swift/cfgs/bebop_site.sh`

For the parameters that specify file paths, `$EMEWS_PROJECT_ROOT` can be used to refer to the `swift_proj` directory in a machine and filesystem independent way.

### Workflow Algorithm Configuration

The workflow algorithm file configures the workflow itself. This file is specified in the job configuration.
See `swift_proj/data/algo_params/test_params.json` for an example. The coniguration parameters are:

* run_type - specifies the type of workflow run to execute. Valid values are "calibration" and "sampleset". The first of these will perform a calibration workflow, calling `PolarisOpt.F.calibrate_simulation`. The second will produce a sampleset, callling `PolarisOpt.F.build_sampleset`.
* settings_file - the path to the settings file (the file defining the scenario control parameters).
* config_file - the path to the config file (the file defining the POLARIS calibration variables).
* num_samples - if the *run_type* is 'sampleset', the number of samples to generate. Otherwise, this
is ignored.
* dr_model_file - the path to the pickled DR model to use during calibration.
* m_model_file - the path to the pickled mean NN to use during calibration.
* quiet - the value to pass as the *quiet* argument when calling `PolarisOpt.F.calibrate_simulation` during
calibration.

_Note_ that any paths specified within the file are relative to `swift_proj/data`. 

## Workflow Details

When the workflow is launched with the submission script (`bebop_run_drgp.sh`),
and running on the compute nodes, `swift_proj/swift/drgp.swift` is executed.
This eventually calls
`swift_proj/python/drgp.py`. `drgp.py` unpacks the workflow algorithm parameters and performs the 
run defined by the configuration. `drgp.py` also copies the settings file and the config
file into the experiment directory and initialize the `SetupManager` with
the paths of the copied files. The originals are not updated during the course
of the workflow.

When model evaluations are requested via `build_sampleset` or `calibrate_simulation`,
the model evaluation arguments are pickled as a list, and each Swift-t worker process runs
the appropriate number of model evaluations using those arguments. For example, if there are 2 workers
and the calibration requires 10 model evalutations, each worker will run 5 model evaluations.
The first worker will use evaluation arguments 1 - 5, and the second worker arguments 6 - 10. 
The workers will run in parallel but the model evaluations on each worker will execute 
sequentially. The python code that launches each model evaluation (i.e., `eval_sim.eval_DR_task` and `eval_sim.eval_sample_task`) runs its own python interpreter where that interpreter,
PYTHONPATH and number of polaris threads are specified in the site file (`swift_proj/swift/cfgs/bebop_site.sh`).


