#!/bin/sh

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
cd /global/project/projectdirs/m2755/GASpy_dev/GASpy_regressions/
python update_model.py >> update_model.log

# CommonAdapter (SLURM) completed writing Template
