#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --partition=regular
#SBATCH --job-name=predict
#SBATCH --output=predict-%j.out
#SBATCH --error=predict-%j.error
#SBATCH --constraint=haswell
#SBATCH --qos=premium

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Use a surrogate model to make predictions
python -c "from gaspy_regress.perform import prediction; prediction()"
