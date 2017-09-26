# GASpy Regressions

## Purpose
[GASpy](https://github.com/ktran9891/GASpy) is able to create various catalyst-adsorbate systems and then
use DFT to simulate the adsorption energies of these systems. This repository, which is meant to be a submodule of
GASpy, analyzes GASpy's results in order to
1. Create surrogate models to identify potentially high-performing catalysts
2. Analyze/visualize GASpy's DFT data and the surrogate models' data

## Overview
The main bread and butter of this submodule is the `GAspyRegressor` class inside the `regressor.py` module.
You instantiate this class by passing it filter settings for it to pull and parse GASdb data from our Mongo DB.
You then use one of the `GASpyRegressor.fit_*` methods to perform the regression. Afterwards, you may pass parsed
mongo documents to the `GASpy_regressor.predict` method to make predictions. You can get these parsed mongo
documents from the `gaspy.utils.get_docs` function.

These models may be combined with GASpy_predict (https://github.com/ktran9891/GASpy_predict) and GASpy
to create automated feedback loops of DFT-simulation, surrogate model regression, prediction, and more DFT-simulation.
To facilitate this feedback loop, we created `update_model.ipynb`. This notebook is a trimmed version of `sandbox.ipynb`
that creates and saves only the models that we plan to use in the feedback loop. We then use Jupyter to convert
`update_model.ipynb` into a standard Python file, `update_model.py`, which can be executed by shell scripts
(`update_model.sh`), which can be executed in Cron jobs. Thus, we can automate the modeling portion of the feedback loop.

`Volcano_plots.ipynb` is the file that analyzes the GASpy and GASpy_regression results for scientific analysis.
