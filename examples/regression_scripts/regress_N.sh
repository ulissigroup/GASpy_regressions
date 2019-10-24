#!/bin/bash


#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=regress_N
#SBATCH --chdir=/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/logs/regressions
#SBATCH --output=regress_N.log
#SBATCH --error=regress_N.log
#SBATCH --open-mode=append
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=FAIL

# OMP has some bug that needs this flag, now
export KMP_INIT_AT_FORK=FALSE

srun shifter python -c "from gaspy_regress import fit_model0_adsorption_energies; fit_model0_adsorption_energies('N')"
