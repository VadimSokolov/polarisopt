#! /usr/bin/env bash

set -eu

if [ "$#" -ne 2 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} exp_id cfg_file"
  exit 1
fi

export TURBINE_LOG=0 TURBINE_DEBUG=0 ADLB_DEBUG=0
# export TURBINE_STDOUT=out-%%r.txt
export TURBINE_STDOUT=
export ADLB_TRACE=0
export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
# source some utility functions used by EMEWS in this script                                                                                 
source "${EMEWS_PROJECT_ROOT}/etc/emews_utils.sh"

export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID
check_directory_exists

CFG_FILE=$2
source $CFG_FILE

echo "--------------------------"
echo "WALLTIME:              $CFG_WALLTIME"
echo "WORKER POOL PROCS:     $CFG_WORKER_POOL_NODES"
echo "WORKER POOL PPN:       $CFG_WORKER_POOL_PPN"
echo "DRGP POOL PROCS:       $CFG_DRGP_NODES"
echo "DRGP POOL PPN:         $CFG_DRGP_PPN"
echo "ALGO PARAM FILE:       $CFG_ALGO_PARAMS_FILE"
echo "DB_HOST:               $CFG_DB_HOST"
echo "DB_USER:               $CFG_DB_USER"
echo "--------------------------"

export QUEUE=$CFG_QUEUE
export WALLTIME=$CFG_WALLTIME
export TURBINE_JOBNAME="${EXPID}_job"
export PROJECT=$CFG_PROJECT

export WORKER_POOL_NODES=$CFG_WORKER_POOL_NODES
export WORKER_POOL_PPN=$CFG_WORKER_POOL_PPN

export DB_HOST=$CFG_DB_HOST
export DB_USER=$CFG_DB_USER
export DB_PORT=$CFG_DB_PORT

export SITE_FILE=$CFG_SITE_FILE

# if R cannot be found, then these will need to be
# uncommented and set correctly.
# export R_HOME=/path/to/R
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib
# if python packages can't be found, then uncommited and set this
# PYTHONPATH="/lcrc/project/EMEWS/bebop/repos/probabilistic-sensitivity-analysis:"
# PYTHONPATH+="/lcrc/project/EMEWS/bebop/repos/panmodel-0.20.0:"
# PYTHONPATH+="$EMEWS_PROJECT_ROOT/python"
# export PYTHONPATH
# echo "PYTHONPATH: $PYTHONPATH"
EQ_SQL_ROOT=/lcrc/project/POLARIS/bebop/sfw/EQ-SQL
export EQ_SQL=$EQ_SQL_ROOT/swift-t/ext
export EQ_SQL_PY=$EQ_SQL_ROOT/python
export POLARIS_OPT=$( readlink --canonicalize $EMEWS_PROJECT_ROOT/../DRGP )
# export PYTHONPATH=$EQ_SQL:$EQ_SQL_PY:$POLARIS_OPT:$EMEWS_PROJECT_ROOT/python
export PYTHONPATH=$EQ_SQL:$EQ_SQL_PY:$POLARIS_OPT:$EMEWS_PROJECT_ROOT/python
echo "PYTHONPATH: $PYTHONPATH"

$EMEWS_PROJECT_ROOT/swift/check-queues.sh

# export SITE=bebop

# Resident task workers and ranks
# export TURBINE_RESIDENT_WORK_WORKERS=1
# export RESIDENT_WORK_RANKS=$(( PROCS - 2 ))

# EQ/R location
# EQR=/lcrc/project/EMEWS/bebop/repos/spack/opt/spack/linux-centos7-broadwell/gcc-7.1.0/eqr-1.0-5hb4aszbbtezlifks6fz4g24zldnkdbx
EMEWS_EXT=$EMEWS_PROJECT_ROOT/ext/emews

# ME VARS ETC
export ME_NODES=$CFG_DRGP_NODES
export ME_PPN=$CFG_DRGP_PPN
export ME_TIMEOUT=$CFG_DRGP_TIMEOUT
export ME_COMMAND=$CFG_DRGP_CMD

mkdir -p $TURBINE_OUTPUT/simulator/Target
mkdir -p $TURBINE_OUTPUT/data/Models

ALGO_PARAMS=$CFG_ALGO_PARAMS_FILE
cp $EMEWS_PROJECT_ROOT/data/$ALGO_PARAMS $TURBINE_OUTPUT/algo_params.json

ME_EXPORTS="PYTHONPATH=$EMEWS_PROJECT_ROOT/python:$EMEWS_PROJECT_ROOT/../DRGP:$EQ_SQL_PY\n"
# ME_EXPORTS+="PYTHONHOME=/soft/anaconda3/2020.11"
ME_EXPORTS+="PYTHONHOME=/home/vsokolov/.conda/envs/pol"
export ME_EXPORTS


# set machine to your schedule type (e.g. pbs, slurm, cobalt etc.),
# or empty for an immediate non-queued unscheduled run
MACHINE="slurm-multijob"
TOTAL_PROCS=$(( WORKER_POOL_NODES * WORKER_POOL_PPN + ME_NODES * ME_PPN ))

# Set these to worker pool values because the worker pool
# runs under turbine and it needs these.
export PPN=$WORKER_POOL_PPN
export PROCS=$(( WORKER_POOL_NODES * WORKER_POOL_PPN ))

if [ -n "$MACHINE" ]; then
  MACHINE="-m $MACHINE"
fi

mkdir -p $TURBINE_OUTPUT/tmp
cp $CFG_FILE $TURBINE_OUTPUT/cfg.sh


CMD_LINE_ARGS="$*"


# Add any script variables that you want to log as
# part of the experiment meta data to the USER_VARS array,
# for example, USER_VARS=("VAR_1" "VAR_2")
USER_VARS=( )
# log variables and script to to TURBINE_OUTPUT directory

export TURBINE_LAUNCHER=srun

PG_LIB=/lcrc/project/EMEWS/bebop/sfw/gcc-7.1.0/postgres-14.2/lib
# MKL=/lcrc/project/EMEWS/bebop/repos/spack/opt/spack/linux-centos7-broadwell/gcc-7.1.0/intel-mkl-2020.1.217-dqzfemzfucvgn2wdx7efg4swwp6zs7ww
# MKL_LIB=$MKL/mkl/lib/intel64
# MKL_OMP_LIB=$MKL/lib/intel64
# LDP=$MKL_LIB/libmkl_def.so:$MKL_LIB/libmkl_avx2.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_lp64.so:$MKL_LIB/libmkl_intel_thread.so:$MKL_OMP_LIB/libiomp5.so

# To avoid: EXCEPTION: /home/nick/.venv/py3.8/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1: cannot allocate memory in static TLS block
# LDP+=:/home/nick/.venv/py3.8/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1

log_script

# echo's anything following this standard out
# set -x

# PROCS is worker pool procs
swift-t -n $PROCS $MACHINE -p \
    -r $EQ_SQL -I $EQ_SQL \
    -I $EMEWS_EXT \
    -e EMEWS_PROJECT_ROOT \
    -e TURBINE_OUTPUT \
    -e TURBINE_LOG \
    -e TURBINE_DEBUG \
    -e ADLB_DEBUG \
    -e DB_HOST \
    -e DB_USER \
    -e DB_PORT \
    -e PYTHONPATH \
    -e LD_LIBRARY_PATH=$PG_LIB:$LD_LIBRARY_PATH \
    -e SITE_FILE \
    -e ME_TIMEOUT \
    $EMEWS_PROJECT_ROOT/swift/worker_pool.swift $CMD_LINE_ARGS

chmod g+rw $TURBINE_OUTPUT/*.tic
