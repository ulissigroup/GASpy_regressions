#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=gdrive_dump
#SBATCH --output=gdrive_dump-%j.out
#SBATCH --error=gdrive_dump-%j.error
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

# Dump our predictions into Google spreadsheets
# CO2RR
python gdrive_dump.py \
    --gsheet gasdb_predictions \
    --worksheet CO2RR_T \
    --predictions /global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl \
    --comparison_predictions /global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl \
    --identifying_labels 'MPID, Miller, Top?' \
    --reporting_labels 'dE [eV], Performance, Mongo ID' \
    --comparison_name HER
# HER
python gdrive_dump.py \
    --gsheet gasdb_predictions \
    --worksheet HER_T \
    --predictions /global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl
