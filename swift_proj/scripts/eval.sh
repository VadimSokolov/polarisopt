#!/bin/bash

# Runs the DGRP evalution functions as passed from the ME
set -eu

FUNC="$1"
PROXIES="$2"
PARAMS="$3"

# site specific file, set via export
if [[ ${SITE_FILE:-} != "" ]]
then
    source $SITE_FILE
fi

EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )

python3 $EMEWS_PROJECT_ROOT/python/eval_wrapper.py "$FUNC" "$PROXIES" "$PARAMS"
