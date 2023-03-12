#!/bin/sh
set -eu

# CHECK QUEUES SH
# Check that the EMEWS Queues are empty before running a test
export LD_LIBRARY_PATH=/lcrc/project/EMEWS/bebop/sfw/gcc-7.1.0/postgres-14.2/lib:$LD_LIBRARY_PATH
THIS=$(   readlink --canonicalize $( dirname $0 ) )
python3 $THIS/../python/check-queues.py

