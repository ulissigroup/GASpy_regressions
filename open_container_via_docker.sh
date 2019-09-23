#!/bin/sh
# This script will use Docker to start an interactive container to run GASpy_regressions.
# Note that you need to run this script from the directory that it is in,
# and it needs to be inside GASpy.

# Optional input argument. If "jupyter", then open a container to run Jupyter.
jupyter=${1:-0}


# Establish out how to mount GASpy to the container. This is the part
# that assumes that you are running this script inside GASpy_regressions,
# which should be inside GASpy
gaspy_regressions_path=$(pwd)
gaspy_path="$(dirname "$gaspy_regressions_path")"
gaspy_mounting_config="$gaspy_path:/home/GASpy"

# Create a container from the image
#   -it     run interactively
#   --rm    close the container when we exit
#   -p      connect to the default port used by Jupyter
#   -v      mount various things to the container
if [ $jupyter = "jupyter" ]; then
    docker run -it --rm -w "/home/GASpy" \
        -p 8888:8888 \
        -v $gaspy_mounting_config \
        -v $HOME/.ssh:/home/.ssh \
        ulissigroup/gaspy_regressions:latest \
        jupyter
else
    docker run -it --rm -w "/home/GASpy" \
        -v $gaspy_mounting_config \
        -v $HOME/.ssh:/home/.ssh \
        ulissigroup/gaspy_regressions:latest \
        /bin/bash
fi
