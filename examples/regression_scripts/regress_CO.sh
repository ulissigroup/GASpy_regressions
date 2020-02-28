#!/bin/bash


#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --account=m2755
#SBATCH --qos=regular
#SBATCH --image=ulissigroup/gaspy_regressions:latest
#SBATCH --volume=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy:/home/GASpy
#SBATCH --job-name=regress_CO
#SBATCH --chdir=/global/cfs/cdirs/m2755/GASpy_workspaces/GASpy/logs/regressions
#SBATCH --output=regress_CO.log
#SBATCH --error=regress_CO.log
#SBATCH --open-mode=append
#SBATCH --mail-user=apalizha@andrew.cmu.edu
#SBATCH --mail-type=FAIL

# OMP has some bug that needs this flag, now
export KMP_INIT_AT_FORK=FALSE

srun shifter python -c "from gaspy_regress import fit_model0_adsorption_energies; fit_model0_adsorption_energies('CO')"
