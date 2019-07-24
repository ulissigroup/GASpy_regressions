#!/bin/bash


#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/jovyan/GASpy
#SBATCH --job-name=regress
#SBATCH --chdir=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/logs
#SBATCH --output=regressions.log
#SBATCH --error=regressions.log
#SBATCH --open-mode=append

srun shifter python -c "from gaspy_regress import fit_model0_adsorption_energies; fit_model0_adsorption_energies('H')"
