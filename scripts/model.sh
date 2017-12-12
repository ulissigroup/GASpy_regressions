#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=regular
#SBATCH --job-name=model
#SBATCH --output=model-%j.out
#SBATCH --error=model-%j.error
#SBATCH --constraint=haswell

# Create and save a surrogate model
python -c "from gaspy_regress.perform import modeling; modeling()"
