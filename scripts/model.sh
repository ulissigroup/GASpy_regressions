#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=regular
#SBATCH --job-name=model
#SBATCH --output=model-%j.out
#SBATCH --error=model-%j.error
#SBATCH --constraint=haswell

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"

# Load the appropriate environment, etc.
module load python
cd GASpy_regressions/scripts
source activate $conda_path

# Create and save a surrogate model
python model.py
