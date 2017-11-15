#!/bin/sh

module load python
source activate /project/projectdirs/m2755/GASpy_conda/
cd /global/project/projectdirs/m2755/GASpy/GASpy_regressions/
python ./automodel/predict.py >> predict.log 2>&1
