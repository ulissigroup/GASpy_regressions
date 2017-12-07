#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=regular
#SBATCH --job-name=model
#SBATCH --output=model-%j.out
#SBATCH --error=model-%j.error
#SBATCH --constraint=haswell

# Load GASpy
. ~/GASpy/scripts/load_env.sh
cd $GASPY_REG_PATH/scripts

# Create and save a surrogate model
python model.py
