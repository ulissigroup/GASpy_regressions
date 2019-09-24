#!/bin/sh -l


#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --account=m2755
#SBATCH --qos=low
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=push
#SBATCH --chdir=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/logs
#SBATCH --output=regressions.log
#SBATCH --error=regressions.log
#SBATCH --open-mode=append

shifter \
    --image=ulissigroup/gaspy_regressions:latest \
    --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy \
    python -c "from gaspy_regress import save_predictions; save_predictions()"
