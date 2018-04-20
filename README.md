# GASpy Regressions

## Purpose
[GASpy](https://github.com/ktran9891/GASpy) is able to create various
catalyst-adsorbate systems and then use DFT to simulate the adsorption energies
of these systems. This repository, which is meant to be a submodule of GASpy,
analyzes GASpy's results in order to

1. Create surrogate models to identify potentially high-performing catalysts
2. Analyze/visualize GASpy's DFT data and the surrogate models' data

## Overview
The main bread and butter of this submodule is the `GAspyRegressor` class
inside the `regressor.py` module. You instantiate this class by passing it
filter settings for it to pull and parse GASdb data from our Mongo DB. You then
use one of the `GASpyRegressor.fit_*` methods to perform the regression.
Afterwards, you may pass parsed mongo documents to the
`GASpy_regressor.predict` method to make predictions. You can get these parsed
mongo documents from the `gaspy.gasdb.get_docs` function.

These models may be combined with
[GASpy_predict](https://github.com/ktran9891/GASpy_predict) and GASpy to create
automated, active learing loops of DFT-simulation, surrogate model regression,
prediction, and more DFT-simulation. To facilitate this active learning, we
create models by executing the `scripts/model.sh` script to create a surrogate
model and then we execute the `scripts/predict.sh` script to make
ML-predictions on our catalog of sites using that model. To keep things
up-to-date and "online", we execute the `model.sh` and `predict.sh` scripts
daily via cron. For illustrative purposes, the `modeling.ipynb` Jupyter
notebook in the `notebooks` folder shows how exactly we perform the regression.

We also have a couple of other files in the `notebooks` folder that show how we
do some other analyses on our regression methods and database of energies.

## Installation
Remember to add the repo to your Python path. The module importing assumes that
you have GASpy in your Python path. You can do so by putting the following in
your `.bashrc`:
```
export PYTHONPATH="/path/to/GASpy/GASpy_regressions:${PYTHONPATH}"
```

In addition to the packages that GASpy requires, you will need the following
Python packages installed:
```
_nb_ext_conf              0.3.0
bokeh                     0.12.10
datashader                0.6.2
datashape                 0.5.4
ipaddress                 1.0.19
ipycache                  0.1.4
ipykernel                 4.6.1
ipyparallel               6.0.2
ipython                   5.5.0
ipython_genutils          0.2.0
ipywidgets                6.0.0
jupyter-console           5.2.0
jupyter_client            5.0.1
jupyter_contrib_core      0.3.3
jupyter_contrib_nbextensions 0.4.0
jupyter_core              4.3.0
jupyter_highlight_selected_word 0.1.0
jupyter_latex_envs        1.4.0
jupyter_nbextensions_configurator 0.4.0
matplotlib                2.0.0
nb_anacondacloud          1.2.0
nb_conda                  2.0.0
nb_conda_kernels          2.0.0
nbconvert                 5.1.1
nbformat                  4.3.0
nbpresent                 3.0.2
notebook                  5.0.0
plotly                    2.2.2
scikit-learn              0.19.0
seaborn                   0.8.1
tpot                      0.9.1
TPOT                      0.8.3
```
