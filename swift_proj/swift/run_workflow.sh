#! /usr/bin/env bash

set -eu

if [ "$#" -ne 2 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} exp_id cfg_file"
  exit 1
fi

# uncomment to turn on swift/t logging. Can also set TURBINE_LOG,
# TURBINE_DEBUG, and ADLB_DEBUG to 0 to turn off logging
# export TURBINE_LOG=1 TURBINE_DEBUG=1 ADLB_DEBUG=1
export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
# source some utility functions used by EMEWS in this script
source "${EMEWS_PROJECT_ROOT}/etc/emews_utils.sh"

export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID
check_directory_exists

CFG_FILE=$2
source $CFG_FILE

echo "--------------------------"
echo "WALLTIME:            $CFG_WALLTIME"
echo "PROCS:               $CFG_PROCS"
echo "PPN                  $CFG_PPN"
echo "QUEUE:               $CFG_QUEUE"
echo "ALGO_PARAMS          $CFG_ALGO_PARAMS"
echo "SITE_FILE            $CFG_SITE_FILE"
echo "--------------------------"

# TODO edit the number of processes as required.
export PROCS=$CFG_PROCS

# TODO edit QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME
# as required. Note that QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME will
# be ignored if MACHINE flag (see below) is not set
export QUEUE=$CFG_QUEUE
export WALLTIME=$CFG_WALLTIME
export PPN=$CFG_PPN
export TURBINE_JOBNAME="${EXPID}_job"


# if R cannot be found, then these will need to be
# uncommented and set correctly.
# export R_HOME=/path/to/R
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib


# Resident task workers and ranks
export TURBINE_RESIDENT_WORK_WORKERS=1
export RESIDENT_WORK_RANKS=$(( PROCS - 2 ))

export SITE_FILE=$CFG_SITE_FILE


# Uncomment this for the BG/Q:
#export MODE=BGQ QUEUE=default

# set machine to your schedule type (e.g. pbs, slurm, cobalt etc.),
# or empty for an immediate non-queued unscheduled run
MACHINE=""

if [ -n "$MACHINE" ]; then
  MACHINE="-m $MACHINE"
fi


mkdir -p $TURBINE_OUTPUT/data
ALGO_PARAMS=$CFG_ALGO_PARAMS
cp $ALGO_PARAMS $TURBINE_OUTPUT/algo_params.json

# EQ/Py location
EQPY=$EMEWS_PROJECT_ROOT/ext/eqpy

export PYTHONPATH=$EMEWS_PROJECT_ROOT/python:$EQPY:$EMEWS_PROJECT_ROOT/../DRGP

CMD_LINE_ARGS="-algo_params_file=$TURBINE_OUTPUT/algo_params.json $*"

# Add any script variables that you want to log as
# part of the experiment meta data to the USER_VARS array,
# for example, USER_VARS=("VAR_1" "VAR_2")
USER_VARS=()
# log variables and script to to TURBINE_OUTPUT directory
log_script

# echo's anything following this to standard out
cd $TURBINE_OUTPUT
set -x
swift-t -n $PROCS $MACHINE -p -I $EQPY -r $EQPY \
    -e TURBINE_OUTPUT \
    -e TURBINE_LOG \
    -e TURBINE_DEBUG \
    -e ADLB_DEBUG \
    -e SITE_FILE \
    -e EMEWS_PROJECT_ROOT \
    $EMEWS_PROJECT_ROOT/swift/drgp.swift \
    $CMD_LINE_ARGS
