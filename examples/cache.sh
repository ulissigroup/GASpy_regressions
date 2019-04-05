#!/bin/sh -l

# Numpy tries to use multiple threads to speed things up, but if we're
# multithreading already then it actually makes things worse. These commands
# will stop numpy from trying to multithread.
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -c 'from gaspy_regress import cache_predictions; cache_predictions()'
