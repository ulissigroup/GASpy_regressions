#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=regular
#SBATCH --account=m2755
#SBATCH --job-name=tpot
#SBATCH --output=tpot-%j.out
#SBATCH --error=tpot-%j.error
#SBATCH --constraint=knl,quad,cache

module load python
source activate /project/projectdirs/m2755/GASpy_conda/
cd /global/project/projectdirs/m2755/Kevin/GASpy/GASpy_regressions/TPOT

python tpot_regression.py

# CommonAdapter (SLURM) completed writing Template
