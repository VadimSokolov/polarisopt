#! /usr/bin/env bash
set -eu

if [ "$#" -ne 2 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} EXPERIMENT_ID SCENARIO_DIR (e.g. ${script_name} experiment_1 test_city)"
  exit 1
fi

PATH=/lcrc/project/EMEWS/bebop/sfw/swift-t-12202018/stc/bin/:$PATH

# this swift-t was used for hepcep
#PATH=/lcrc/project/MRSA/bebop/sfw/swift-t-38569e3/stc/bin:$PATH

export PATH

# uncomment to turn on swift/t logging. Can also set TURBINE_LOG,
# TURBINE_DEBUG, and ADLB_DEBUG to 0 to turn off logging
# export TURBINE_LOG=1 TURBINE_DEBUG=1 ADLB_DEBUG=1
THIS=$( cd $( dirname $0 ) ; /bin/pwd )
export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
# source some utility functions used by EMEWS in this script
source "${EMEWS_PROJECT_ROOT}/etc/emews_utils.sh"

export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID
check_directory_exists

# TODO edit the number of processes as required.
export PROCS=12

# TODO edit QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME
# as required. Note that QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME will
# be ignored if the MACHINE variable (see below) is not set.
export QUEUE=bdwall
export WALLTIME=00:5:00
export PPN=36
export TURBINE_JOBNAME="${EXPID}_job"

# if R cannot be found, then these will need to be
# uncommented and set correctly.
# export R_HOME=/path/to/R
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib
# if python packages can't be found, then uncommited and set this
# export PYTHONPATH=/path/to/python/packages
export PYTHONHOME=/soft/anaconda3/5.2.0/lib/python3.6
export PYTHONPATH=$EMEWS_PROJECT_ROOT/python:$EMEWS_PROJECT_ROOT/ext/EQ-Py:/soft/anaconda3/5.2.0/lib/python3.6/site-packages:/soft/anaconda3/5.2.0/lib/python3.6:/soft/anaconda3/5.2.0/lib/python3.6/lib-dynload

# TODO edit command line arguments as appropriate
# for your run. Note that the default $* will pass all of this script's
# command line arguments to the swift script.
CMD_LINE_ARGS="$*"

SCENARIO_DIR=$2

# set machine to your schedule type (e.g. pbs, slurm, cobalt etc.),
# or empty for an immediate non-queued unscheduled run
MACHINE="slurm"

if [ -n "$MACHINE" ]; then
  MACHINE="-m $MACHINE"
fi

# Add any script variables that you want to log as
# part of the experiment meta data to the USER_VARS array,
# for example, USER_VARS=("VAR_1" "VAR_2")
USER_VARS=()
# log variables and script to to TURBINE_OUTPUT directory
log_script

# echo's anything following this standard out
set -x

swift-t -n $PROCS $MACHINE -p  \
  $EMEWS_PROJECT_ROOT/swift/polaris_sweep.swift \
  -f="$EMEWS_PROJECT_ROOT/data/upf.txt" \
  -scenario_dir=$SCENARIO_DIR \
   $CMD_LINE_ARGS
