#!/bin/bash


#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=m2755
#SBATCH --qos=premium
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/jovyan/GASpy
#SBATCH --job-name=predict
#SBATCH --chdir=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/logs
#SBATCH --output=predictions.log
#SBATCH --error=predictions.log
#SBATCH --open-mode=append

# Numpy tries to use multiple threads to speed things up, but if we're
# multithreading already then it actually makes things worse. The export
# commands will stop numpy from trying to multithread.
srun shifter bash -c 'export MKL_NUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; export OMP_NUM_THREADS=1; python -c "from gaspy_regress import cache_predictions; cache_predictions(processes=68)"'
