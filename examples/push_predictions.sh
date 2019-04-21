#!/bin/sh -l


# Find the cache of predictions
gasdb_dir=$(python -c "from gaspy.utils import read_rc; print(read_rc('gasdb_path'))")
prediction_cache="predictions.pkl"
predictions_location="$gasdb_dir$prediction_cache"

# Only push the predictions if they are less than a day old
cache_age=$(( ($(date +%s) - $(date +%s -r $predictions_location)) / 86400 ))
if [ $cache_age -eq "0" ]; then
    shifter \
        --image=ulissigroup/gaspy_regressions:latest \
        --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/jovyan/GASpy \
        python -c "from gaspy_regress import save_predictions; save_predictions()"
fi
