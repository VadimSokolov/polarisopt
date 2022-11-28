export PATH=/soft/anaconda3/2020.11/bin/:$PATH
export PYTHONPATH=$EMEWS_PROJECT_ROOT/python:$EMEWS_PROJECT_ROOT/ext/EQ-SQL:$EMEWS_PROJECT_ROOT/../DRGP

# these env vars should be set in the bebop_submit.sh script
export PYTHONPATH=$EQ_SQL:$EQ_SQL_PY:$POLARIS_OPT:$EMEWS_PROJECT_ROOT/python

export POLARIS_NUM_THREADS=$(( 36 / $WORKER_POOL_PPN ))