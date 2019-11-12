#!/bin/sh -l


# Only do things if we don't have a regression job already running
if ! squeue -u ktran | grep -E 'regress|predict|push'; then

    # Perform the regressions on our DFT data
    regress_C=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_C.sh)
    regress_CO=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_CO.sh)
    regress_H=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_H.sh)
    regress_N=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_N.sh)
    regress_O=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_O.sh)
    regress_OH=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_OH.sh)
    #regress_OOH=$(sbatch --parsable /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_OOH.sh)

    # Perform ML predictions on our catalog
    predict_C=$(sbatch --parsable --dependency=afterok:$regress_C /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_C.sh)
    predict_CO=$(sbatch --parsable --dependency=afterok:$regress_CO /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_CO.sh)
    predict_H=$(sbatch --parsable --dependency=afterok:$regress_H /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_H.sh)
    predict_N=$(sbatch --parsable --dependency=afterok:$regress_N /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_N.sh)
    predict_O=$(sbatch --parsable --dependency=afterok:$regress_O /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_O.sh)
    predict_OH=$(sbatch --parsable --dependency=afterok:$regress_OH /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_OH.sh)
    #predict_OOH=$(sbatch --parsable --dependency=afterok:$regress_OOH /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_OOH.sh)

    # Push the predictions into Mongo
    push=$(sbatch --parsable --dependency=afterany:$predict_C,afterany:$predict_CO,afterany:$predict_H,afterany:$predict_N,afterany:$predict_O,afterany:$predict_OH /global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/push_predictions.sh)
fi
