#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --account=m2755
#SBATCH --job-name=alamo
#SBATCH --output=alamo-%j.out
#SBATCH --error=alamo-%j.error
#SBATCH --constraint=knl,quad,cache

module load python
source activate /project/projectdirs/m2755/GASpy_conda/
cd /global/project/projectdirs/m2755/Kevin/GASpy/GASpy_regressions/

python regress.py > regression.log

# CommonAdapter (SLURM) completed writing Template
