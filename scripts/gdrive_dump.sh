#!/bin/sh

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"

# Load the appropriate environment, etc.
module load python
cd GASpy_regressions/scripts
source activate $conda_path

# Dump our predictions into Google spreadsheets
python gdrive_dump.py gasdb_predictions CO2RR_T CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl
python gdrive_dump.py gasdb_predictions HER_T HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl
