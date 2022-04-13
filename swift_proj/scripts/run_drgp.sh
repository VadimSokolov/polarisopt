set -eu

if [ "$#" -ne 2 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} exp_id cfg_file"
  exit 1
fi

export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID
mkdir -p $TURBINE_OUTPUT

CFG_FILE=$2
source $CFG_FILE

export DB_HOST=$CFG_DB_HOST
export DB_USER=$CFG_DB_USER
export DB_PORT=$CFG_DB_PORT
export DB_NAME=$CFG_DB_NAME

export PYTHONPATH=$EMEWS_PROJECT_ROOT/ext/EQ-SQL:$EMEWS_PROJECT_ROOT/../DRGP 
echo $PYTHONPATH

ALGO_PARAMS=$CFG_ALGO_PARAMS_FILE
cp $EMEWS_PROJECT_ROOT/data/$ALGO_PARAMS $TURBINE_OUTPUT/algo_params.json 

# TODO replace this with copying Polaris files???
mkdir -p $TURBINE_OUTPUT/simulator/Target
mkdir -p $TURBINE_OUTPUT/data/Models
touch $TURBINE_OUTPUT/simulator/Target/bloomington-Result.sqlite

python $EMEWS_PROJECT_ROOT/python/drgp.py $EXPID $TURBINE_OUTPUT $TURBINE_OUTPUT/algo_params.json 
