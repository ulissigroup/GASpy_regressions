#!/bin/bash


#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=predict_N
#SBATCH --chdir=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/logs/regressions
#SBATCH --output=predict_N.log
#SBATCH --error=predict_N.log
#SBATCH --open-mode=append
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=FAIL

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun shifter python -c "from gaspy_regress import cache_predictions; cache_predictions('N', processes=64)"
