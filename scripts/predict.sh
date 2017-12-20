#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --job-name=predict
#SBATCH --output=predict-%j.out
#SBATCH --error=predict-%j.error
#SBATCH --constraint=haswell

# Use a surrogate model to make predictions
python -c "from gaspy_regress.perform import prediction; prediction()"
