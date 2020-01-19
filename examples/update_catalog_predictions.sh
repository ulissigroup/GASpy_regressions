#!/bin/sh -l


# Only do things if we don't have a regression job already running
if ! squeue -u ktran | grep -E 'regress|predict|push'; then

    # Initialize the dependencies for pushing to the catalog
    dependencies=""

    # C
    regress_C=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_C.sh)
    predict_C=$(sbatch --parsable --dependency=afterok:$regress_C /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_C.sh)
    dependencies="$dependencies,afterany:$predict_C"

    # CO
    regress_CO=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_CO.sh)
    predict_CO=$(sbatch --parsable --dependency=afterok:$regress_CO /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_CO.sh)
    dependencies="$dependencies,afterany:$predict_CO"

    # H
    regress_H=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_H.sh)
    predict_H=$(sbatch --parsable --dependency=afterok:$regress_H /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_H.sh)
    dependencies="$dependencies,afterany:$predict_H"

    # N
    regress_N=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_N.sh)
    predict_N=$(sbatch --parsable --dependency=afterok:$regress_N /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_N.sh)
    dependencies="$dependencies,afterany:$predict_N"

    # O
    regress_O=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_O.sh)
    predict_O=$(sbatch --parsable --dependency=afterok:$regress_O /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_O.sh)
    dependencies="$dependencies,afterany:$predict_O"

    # OH
    regress_OH=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_OH.sh)
    predict_OH=$(sbatch --parsable --dependency=afterok:$regress_OH /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_OH.sh)
    dependencies="$dependencies,afterany:$predict_OH"

    # OOH
    regress_OOH=$(sbatch --parsable /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/regress_OOH.sh)
    predict_OOH=$(sbatch --parsable --dependency=afterok:$regress_OOH /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/regression_scripts/predict_OOH.sh)
    dependencies="$dependencies,afterany:$predict_OOH"

    # Push to catalog
    dependencies=${dependencies#","}
    push=$(sbatch --parsable --dependency=$dependencies /global/cfs/cdirs/m2755/GASpy/GASpy_regressions/examples/regression_scripts/push_predictions.sh)
fi
