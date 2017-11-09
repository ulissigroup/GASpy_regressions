#!/bin/sh

module load python
source activate /project/projectdirs/m2755/GASpy_conda/
cd /global/project/projectdirs/m2755/GASpy/GASpy_regressions/
python ./automodel/update_model.py >> update_model.log 2>&1
