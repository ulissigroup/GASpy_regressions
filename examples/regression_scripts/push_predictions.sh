#!/bin/sh -l


#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=push
#SBATCH --chdir=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/logs/regressions
#SBATCH --output=pushing.log
#SBATCH --error=pushing.log
#SBATCH --open-mode=append
#SBATCH --mail-user=apalizha@andrew.cmu.edu
#SBATCH --mail-type=ALL

shifter \
    --image=ulissigroup/gaspy_regressions:latest \
    --volume=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy \
    python -c "from gaspy_regress import save_predictions; save_predictions()"
