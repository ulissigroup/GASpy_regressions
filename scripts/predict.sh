#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=regular
#SBATCH --job-name=predict
#SBATCH --output=predict-%j.out
#SBATCH --error=predict-%j.error
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

# Use a surrogate model to make predictions
python predict.py
