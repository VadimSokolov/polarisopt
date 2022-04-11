# Bebop emews pool site file
module load anaconda3/2020.07

export PYTHONPATH=$EMEWS_PROJECT_ROOT/../DRGP
export POLARIS_NUM_THREADS=$(( 36 / $PPN ))
