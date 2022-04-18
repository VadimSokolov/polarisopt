set -eu

if [ "$#" -lt 2 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} exp_id cfg_file [worker_pool_job_id]"
  exit 1
fi

export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
# source some utility functions used by EMEWS in this script                                                                                 
source "${EMEWS_PROJECT_ROOT}/etc/emews_utils.sh"

export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID


export EXP_DIR=$TURBINE_OUTPUT
CFG_FILE=$2
source $CFG_FILE

export ME_TIMEOUT=$CFG_DRGP_TIMEOUT

mkdir -p $EXP_DIR

export DB_HOST=$CFG_DB_HOST
export DB_PORT=$CFG_DB_PORT
export DB_NAME=$CFG_DB_NAME
export DB_USER=$CFG_DB_USER

export PYTHONPATH=$EMEWS_PROJECT_ROOT/ext/EQ-SQL:$EMEWS_PROJECT_ROOT/../DRGP 
echo "PYTHONPATH: $PYTHONPATH"

ALGO_PARAMS=$CFG_ALGO_PARAMS_FILE
cp $EMEWS_PROJECT_ROOT/data/$ALGO_PARAMS $TURBINE_OUTPUT/algo_params.json 

export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

MKL=/lcrc/project/EMEWS/bebop/repos/spack/opt/spack/linux-centos7-broadwell/gcc-7.1.0/intel-mkl-2020.1.217-dqzfemzfucvgn2wdx7efg4swwp6zs7ww
MKL_LIB=$MKL/mkl/lib/intel64
MKL_OMP_LIB=$MKL/lib/intel64
LDP=$MKL_LIB/libmkl_def.so:$MKL_LIB/libmkl_avx2.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_lp64.so:$MKL_LIB/libmkl_intel_thread.so:$MKL_OMP_LIB/libiomp5.so
PSQL_LIB=/lcrc/project/EMEWS/bebop/sfw/gcc-7.1.0/postgres-14.2/lib

export LD_LIBRARY_PATH=$MKL_LIB:$PSQL_LIB:$LD_LIBRARY_PATH
export LD_PRELOAD=$LDP

# TODO replace this with copying Polaris files???
mkdir -p $TURBINE_OUTPUT/simulator/Target
mkdir -p $TURBINE_OUTPUT/data/Models

NODES=$(( CFG_PROCS / CFG_PPN ))
mkdir -p $EXP_DIR
TEMPLATE=$EMEWS_PROJECT_ROOT/scripts/me_submission_template.sbatch
ME_SLURM=$EXP_DIR/me_slurm.sh
sed -e "s|\${EXP_DIR}|$EXP_DIR|g" \
    -e "s|\${JOB_NAME}|$EXPID|g" \
    -e "s|\${QUEUE}|$CFG_QUEUE|g" \
    -e "s|\${PROJECT}|$CFG_PROJECT|g" \
    -e "s|\${WALLTIME}|$CFG_WALLTIME|g" \
    -e "s|\${NODES}|$NODES|g" \
    -e "s|\${PPN}|$CFG_PPN|g" \
    -e "s|\${PROCS}|$CFG_PROCS|g" \
    -e "s|\${CMD}|$CFG_CMD|g" \
    $TEMPLATE > $ME_SLURM

DEP_CLAUSE=""
if [ "$#" -eq 3 ]; then
  DEP_CLAUSE="--dependency=after:$3"
fi

echo $DEP_CLAUSE
sbatch $DEP_CLAUSE $ME_SLURM