# GASpy Regressions

## Purpose
[GASpy](https://github.com/ktran9891/GASpy) is able to create various catalyst-adsorbate systems and then
use DFT to simulate the adsorption energies of these systems. GASpy_regressions is meant to analyze
GASpy's results in order to
1. Create surrogate models to identify potentially high-performing catalysts
2. Analyze/visualize GASpy's DFT data and the surrogate models' data

## Overview
`pull_features.py` contains a Python class, `PullFeatures`, that pulls and pre-processes data from GASpy's database.
`Regress.ipynb` then uses these pre-processed data to perform regressions, which yield models that
can predict adsorption energies from structural information (e.g., coordination number, adsorbate identity, etc.).

These models may be combined with GASpy_predict (https://github.com/ktran9891/GASpy_predict) and GASpy
to create automated feedback loops of DFT-simulation, surrogate model regression, prediction, and more DFT-simulation.
To facilitate this feedback loop, we created `update_model.ipynb`. This notebook is a trimmed version of `Regress.ipynb`
that creates and saves only the models that we plan to use in the feedback loop. We then use Jupyter to convert
`update_model.ipynb` into a standard Python file, `update_model.py`, which can be executed by shell scripts
(`update_model.sh`), which can be executed in Cron jobs. Thus, we can automate the modeling portion of the feedback loop.

`Volcano_plots.ipynb` is the file that analyzes the GASpy and GASpy_regression results for scientific analysis.
