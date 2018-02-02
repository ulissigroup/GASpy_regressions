#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --job-name=model
#SBATCH --output=model-%j.out
#SBATCH --error=model-%j.error
#SBATCH --constraint=haswell
#SBATCH --qos=premium

# Create and save a surrogate model
python -c "from gaspy_regress.perform import modeling; modeling(fit_blocks=[('CO',), ('H',)])"
