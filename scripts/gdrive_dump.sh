#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=regular
#SBATCH --job-name=gdrive_dump
#SBATCH --output=gdrive_dump-%j.out
#SBATCH --error=gdrive_dump-%j.error
#SBATCH --constraint=haswell

# Dump our predictions into Google spreadsheets

# CO2RR
gsheet="gasdb_predictions"
worksheet="CO2RR_T"
predictions_path="/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl"
comparisons_path="/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl"
identifying_labels="MPID, Miller, Top?"
reporting_labels="dE [eV], Performance, Mongo ID"
comparison_name="HER"
python -c "from gaspy_regress.gio import gdrive_dump; gdrive_dump('$gsheet', '$worksheet', predictions_path='$predictions_path', comparisons_path='$comparisons_path', identifying_labels='$identifying_labels', reporting_labels='$reporting_labels', comparison_name='$comparison_name')"

# HER
gsheet="gasdb_predictions"
worksheet="HER_T"
predictions_path="/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl"
python -c "from gaspy_regress.gio import gdrive_dump; gdrive_dump('$gsheet', '$worksheet', predictions_path='$predictions_path')"
