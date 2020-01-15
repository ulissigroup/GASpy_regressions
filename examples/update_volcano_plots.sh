#!/bin/sh -l

shifter --image=ulissigroup/gaspy_regressions:latest \
    --volume=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy \
    python /global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/examples/update_volcano_plots.py
