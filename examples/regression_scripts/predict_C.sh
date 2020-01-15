#!/bin/bash


#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=30:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=predict_C
#SBATCH --chdir=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/logs/regressions
#SBATCH --output=predict_C.log
#SBATCH --error=predict_C.log
#SBATCH --open-mode=append
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=FAIL

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun shifter python -c "from gaspy_regress import cache_predictions; cache_predictions('C', processes=64)"
