#!/bin/sh -l

gaspy_dir="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy"

# Perform the regressions on our DFT data
salloc -N 1 -C haswell -t 04:00:00 --qos=interactive \
    --image=ulissigroup/gaspy_regressions:latest \
    --volume=$gaspy_dir:/home/jovyan/GASpy \
    < $gaspy_dir/GASpy_regressions/examples/regress.sh

# Make and cache our ML predictions in a pickle file
salloc -N 1 -C haswell -t 04:00:00 --qos=interactive \
    --image=ulissigroup/gaspy_regressions:latest \
    --volume=$gaspy_dir:/home/jovyan/GASpy \
    < $gaspy_dir/GASpy_regressions/examples/cache.sh

# Wait until a certain time of day before we...
difference=$(($(date -d "18:00" +%s) - $(date +%s)))
if [ $difference -lt 0 ]
then
    sleep $((86400 + difference))
else
    sleep $difference
fi
# Save the cache of ML predictions into Mongo
$gaspy_dir/GASpy_regressions/examples/save.sh
shifter \
    --image=ulissigroup/gaspy_regressions:latest \
    --volume=$gaspy_dir:/home/jovyan/GASpy \
    python -c 'from gaspy_regress import save_predictions; save_predictions()'

# Note that we wait until 18:00 PST or 21:00 EST, because it takes about 10
# hours to write the predictions to Mongo. During that time, we cannot use that
# collection efficiently. That's why we wait until the evening to save them, so
# that it'll be less likely that someone is trying to use the collection.
