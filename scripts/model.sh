#!/bin/sh

# Get path information from the .gaspyrc.json file
gaspy_path="$(python ../../.read_rc.py gaspy_path)"
conda_path="$(python ../../.read_rc.py conda_path)"

# Load the appropriate environment, etc.
module load python
cd $gaspy_path/GASpy_regressions/scripts
source activate $conda_path

# Create and save a surrogate model
python model.py >> model.log 2>&1
