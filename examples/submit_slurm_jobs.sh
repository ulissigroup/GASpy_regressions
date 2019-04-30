#!/bin/sh -l


# Only do things if we don't have a regression job already running
if ! squeue -u ktran | grep regress; then

    # Perform the regressions on our DFT data
    job_id_CO=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_CO.sh)
    job_id_H=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_H.sh)
    job_id_N=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_N.sh)
    job_id_O=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_O.sh)
    job_id_OH=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_OH.sh)
    job_id_OOH=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regress_OOH.sh)

    # Create and cache the regression's predictions
    job_id_cache=$(sbatch --parsable --dependency=afterany:$job_id_CO,afterany:$job_id_H,afterany:$job_id_N,afterany:$job_id_O,afterany:$job_id_OH,afterany:$job_id_OOH /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/cache_predictions.sh)
fi
