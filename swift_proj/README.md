# EMEWS Calibration Workflow

The DRGP workflow consists of the DRGP algorithm, a pool
of workers that will execute Polaris, and a database used
to communicate between the two. The DRGP algorithm 
produces input paramters and submits them to the database,
and waits results. The worker pool queries the database
for the input parameters and performs Polaris runs using
those parameters. It then submits the results back to the
database.

The workflow requires some one time setup before it can be
executed.

## One Time Database Setup ##

Each person that runs the workflow needs create their own
database directory. Use the EQ/SQL database utilities to do this.
Those are currently installed in:

`/lcrc/project/POLARIS/bebop/sfw/EQ-SQL/db`

1. Create the data directory. 

```bash
$ cd /lcrc/project/POLARIS/bebop/sfw/EQ-SQL/db
$ source env-polaris-bebop.sh
# The database will be created in the DB_DATA directory
$ export DB_DATA=/lcrc/project/POLARIS/${USER}/emews-db
$ initdb -D $DB_DATA -g
$ cp sample_postgresql.conf $DB_DATA/postgresql.conf
$ cp sample_pg_hba.conf $DB_DATA/pg_hba.conf
```

2. Start the server, create a database within the data directory,
and create the required emews sql tables.

```bash
$ cd EQ-SQL/db
$ source env-polaris-bebop.sh
# Using '/lcrc/project/EMEWS/db/plima' as an example data directory
$ export DB_DATA=/lcrc/project/POLARIS/${USER}/emews-db
$ export DB_NAME=EQ_SQL
# Start the db server
$ ./db-start.sh
$ createdb --host=$DB_HOST --port=$DB_PORT $DB_NAME
# Confirm the database was created
$ ./db-ping.sh
# Create the emews sql tables
$ ./db-mk-tables.sh
# Stop the db server
$ ./db-stop.sh
```

## Starting the Databse ##

Before any workflow that uses the DB is run, the server must be 
started. The scripts in the polaris-hpc/swift_proj/db are used to start and stop
the database.

```bash
# this should be the data directory created in the one time setup
$ export DB_DATA=/lcrc/project/POLARIS/${USER}/emews-db
$ cd polaris-hpc/swift_proj/db
$ source env-polaris-bebop.sh
$ ./db-start.sh
```

When you start the database, the `DB_HOST`, `DB_PORT` and other important environment variables
will be displayd as output. These values will also be saved to a
`db_env_vars_N.txt` file where N is a timestamp in case they are needed later. 
You will need to be logged into DB_HOST to stop the database.

## Stop the Database ##

Currently to stop the database, you *MUST* be logged onto the same login node where the database was
started. So, if in the `db-start.sh` output, you see:

`DB_HOST=beboplogin4.lcrc.anl.gov`

then you must login to `beboplogin4` to stop the database, and do the following:

```bash
$ cd polaris-hpc/swift_proj/db
$ export DB_DATA=/lcrc/project/POLARIS/${USER}/emews-db
$ source env-polaris-bebop.sh
$ ./db-stop.sh
```

Note: If you haven't closed the terminal in which you started the db, then you can just do
`./db-stop.sh`

## Running the Workflow ##

The database server must be started (see above) before running the workflow. 
To run the workflow:

```bash
$ cd polaris-hpc/swift_proj
# Setup the environment
$ source envs/bebop_env.sh
$ cd swift
# X should be an experiment id (e.g, exp_1)
$ ./bebop_submit.sh X ../data/cfgs/bebop_submit.cfg
```

`bebop_submit.sh` submits a single job to the Bebop's slurm scheduler that consists of
launching two executbles. The first
of these starts the DRGP code that produces Polaris input parameters and the second starts
a pool of workers that consumes these parameters and performs the Polaris runs. `bebop_submit.sh`
takes two arguments and experiment id and a configuration file, and creates an
*experiment* directory, `swift_proj/experiments/X` where X is the experiment id,
that contains the workflow output for that job. A `turbine-output` symlink to that experiment
directory is also created in `swift_proj/swift`. Within the experiment directory, the output
is spread among following files. Note that these will not exist untl the job actually begins
running on the compute nodes:

* me_output.txt - contains the output of the DRGP algorithm 
* output.txt - contains the worker pool output
* tmp/err_N|out_N.txt - contains the error and standard output from the indivudal
Polaris model runs. N is a numeric id for a particular run (e.g. err_0.txt and out_0.txt).

The configuration file `polaris-hpc/swift_proj/data/cfgs/bebop_submit.cfg` is used to
configure these two jobs. The configuration file is sourced by `bebop_submit.sh`
and the environment variables defined in the config file are then used to configure
a model run. 

An example configuration file:

```bash
# The expected run duration of the job: corresponds to sbatch's 
# time argument
CFG_WALLTIME=12:00:00
# Bebop queue to submit the job to
CFG_QUEUE=bdwall
# The account to charge the job to
CFG_PROJECT=POLARIS

# Site specific variables etc. used by the worker pool
# when starting a Polaris Run
CFG_SITE_FILE=$EMEWS_PROJECT_ROOT/data/sites/bebop_site.sh

# WORKER POOL CONFIG
# The number of nodes in the worker pool
CFG_WORKER_POOL_NODES=2
# The number of processes per node for the worker pool
# The bebop_site.sh files sets POLARIS_NUM_THREADS to 
# 36 / CFG_WORKER_POOL_PPN
CFG_WORKER_POOL_PPN=36

# DRGP ALGORITHM CONFIG
# There's real reason to change these
# The number of nodes for DRGP
CFG_DRGP_NODES=1
# The number of processes per node for the DRGP algorithm
CFG_DRGP_PPN=1
# json files used to initialize the DRGP algorithm
# Path is relative to swift_proj/data
CFG_ALGO_PARAMS_FILE=algo_params/bebop_test_params.json
# How long, in seconds, each round of polaris runs should take.
# DRGP will wait this long for results from polaris runs before
# throwing an error
CFG_DRGP_TIMEOUT=3600
# The command used to run the DRGP algorithm.
CFG_DRGP_CMD="python3 $EMEWS_PROJECT_ROOT/python/drgp.py $EXPID $TURBINE_OUTPUT $TURBINE_OUTPUT/algo_params.json"

# DATABASE CONFIG
# CFG_DB_HOST and CFG_DB_PORT should be set the current host
# and port as reported when the database was started.
CFG_DB_HOST=beboplogin4.lcrc.anl.gov
CFG_DB_PORT=11219
CFG_DB_NAME=EQ_SQL
CFG_DB_USER=${DB_USER:-$USER}
```

### DRGP Algorithm Configuration

The DRGP algorithm file configures the workflow itself. This file is specified in the job configuration.
See `swift_proj/data/algo_params/bebo_test_params.json` for an example. The coniguration parameters are:

* run_type - specifies the type of workflow run to execute. Valid values are "calibration" and "sampleset".
  The first of these will perform a calibration workflow, calling `PolarisOpt.F.calibrate_simulation`.
  The second will produce a sampleset, callling `PolarisOpt.F.build_sampleset`.
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

When the workflow is launched with the submission script (`bebop_submit.sh`),
the DRGP algorithm begins with `swift_proj/python/drgp.py`. `drgp.py` unpacks the 
workflow algorithm parameters and produces the input parameters as defined by the
configuration. `drgp.py` also copies the settings file and the config
file into the experiment directory and initialize the `SetupManager` with
the paths of the copied files. The originals are not updated during the course
of the workflow.

When polaris model evaluations are requested via `build_sampleset` or `calibrate_simulation`,
those evaluations are run in the worker pool submitted by `bebop_submit.sh` and implemented
in `swift_proj/swift/worker_pool.swift`. The workers will run in parallel. The python code that
launches each model evaluation (i.e., `eval_sim.eval_DR_task` and `eval_sim.eval_sample_task`) runs
its own python interpreter where that interpreter, PYTHONPATH and number of polaris threads are
specified in the site file (`swift_proj/data/sites/bebop_site.sh`).

## DRGP Python Dependencies

The calibration workflow requires additional python packages to be installed.

```
# load the correct Python
module load anaconda3/2020.11
# install pytorch
pip3 install torch --user
# install gpytorch
pip3 install gpytorch --user
# install botorch v.0.1.4
pip3 install botorch==0.1.4 --user
pip3 install dill --user
pip3 install pyDOE --user
```
