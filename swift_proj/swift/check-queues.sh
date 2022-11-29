#!/bin/sh
set -eu

# CHECK QUEUES SH
# Check that the EMEWS Queues are empty before running a test

THIS=$(   readlink --canonicalize $( dirname $0 ) )
python3 $THIS/../python/check-queues.py

