#!/bin/sh -l

# Only do things if we don't have a regression job already running
if ! squeue -u ktran | grep regress; then

    # Perform the regressions on our DFT data
    job_id_regression=$(sbatch --parsable regress.sh)

    # Create and cache the regression's predictions
    job_id_cache=$(sbatch --parsable --dependency=afterany:$job_id_regression cache_predictions.sh)
fi
