
# coding: utf-8

# # regress.ipynb
# Author:  Kevin Tran <ktran@andrew.cmu.edu>
# 
# This python notebook performs regressions on data pulled from a processed mongo DB created by GASpy. It then saves these regressions into pickles (for later use) and creates parity plots of the regression fits.

# # Initialize

# ## Importing

# In[1]:

# Debugging & other Python tools
import pdb
import sys
from pprint import pprint
import itertools
# Saving/loading
import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions (for alamopy)
# Regression
from sklearn.gaussian_process import GaussianProcessRegressor
from tpot import TPOTRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
# Plotting
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
# GASpy
from regression_processor import RegressionProcessor
from pull_features import PullFeatures
sys.path.append('..')
from gaspy.utils import vasp_settings_to_str


# ## Scope

# In[2]:

# Define the feature sets that you want to investigate. They should be
# string names of the PullFeatures methods that you want to use.
FEATURE_SETS = [
                #'energy_fr_coordcount',
                #'energy_fr_coordcount_nncoord',
                'energy_fr_coordcount_ads',
                #'energy_fr_coordcount_nncoord_ads',
                #'energy_fr_nncoord',
                #'energy_fr_gcn_ads',
               ]

# Only pull data that used the following vasp settings
VASP_SETTINGS = vasp_settings_to_str({'gga': 'RP',
                                      'pp_version': '5.4',
                                      'encut': 350})
#VASP_SETTINGS = None

# This is a dictionary that will hold all of the data we need for plotting
DATA = {}


# # Regress

# ## Hierarchical
# TODO:  Test the iterable nature of these cells (i.e., use more than one outer and inner combo)

# ### Execute

# In[ ]:

# Specify the model blocking. Use [] if you don't want blocking (this will help with saving)
#blocks = ['adsorbate']
blocks = []

# Outer regression information
OUTER_FEATURE_SETS = ['energy_fr_coordcount_ads']
OUTER_REGRESSORS = [TPOTRegressor(generations=4,
                                  population_size=16,
                                  verbosity=2,
                                  random_state=42)]
OUTER_REGRESSION_METHODS = ['tpot']
OUTER_SYSTEMS = [(outer_feature_set, OUTER_REGRESSORS[i], OUTER_REGRESSION_METHODS[i])
                 for i, outer_feature_set in enumerate(OUTER_FEATURE_SETS)]
# Inner regression information
INNER_FEATURE_SETS = ['energy_fr_nncoord']
#K = 1.0*RBF(length_scale=1.0) + 1.0*WhiteKernel(noise_level=0.05**2.0) 
K = None
INNER_REGRESSORS = [GaussianProcessRegressor(kernel=K, n_restarts_optimizer=2)]
INNER_REGRESSION_METHODS = ['sk_regressor']
INNER_SYSTEMS = [(inner_feature_set, INNER_REGRESSORS[i], INNER_REGRESSION_METHODS[i])
                 for i, inner_feature_set in enumerate(INNER_FEATURE_SETS)]

# `FEATURE_COMBINATIONS` is a list of tuples for the different combinations
# of the outer and inner regressors we want. We use it to initialize the dictionaries
# of our results.
FEATURE_COMBINATIONS = [combo
                        for combo in itertools.product(*[OUTER_FEATURE_SETS,
                                                         INNER_FEATURE_SETS])]
models = dict.fromkeys(FEATURE_COMBINATIONS)
rmses = dict.fromkeys(FEATURE_COMBINATIONS)
errors = dict.fromkeys(FEATURE_COMBINATIONS)
x = dict.fromkeys(FEATURE_COMBINATIONS)
y = dict.fromkeys(FEATURE_COMBINATIONS)
p_docs = dict.fromkeys(FEATURE_COMBINATIONS)
pp = dict.fromkeys(FEATURE_COMBINATIONS)
block_list = dict.fromkeys(FEATURE_COMBINATIONS)
# Initialize other output dictionaries
RPs = dict.fromkeys(OUTER_FEATURE_SETS)
norm = dict.fromkeys(OUTER_FEATURE_SETS+FEATURE_COMBINATIONS)

# Perform the regressions for each combination of feature sets
for o_feature_set, o_regressor, o_regression_method in OUTER_SYSTEMS:
    # Initialize `RegressionProcessor` to pull the data
    RPs[o_feature_set] = RegressionProcessor(o_feature_set,
                                             blocks=blocks,
                                             vasp_settings=VASP_SETTINGS)
    # Perform the outer regressions
    outer_models, outer_rmses, outer_errors =             getattr(RPs[o_feature_set], o_regression_method)(o_regressor)
    # Perform the inner regressions
    for i_feature_set, i_regressor, i_regression_method in INNER_SYSTEMS:
        models[(o_feature_set, i_feature_set)],             rmses[(o_feature_set, i_feature_set)],             errors[(o_feature_set, i_feature_set)],             _, inner_norm                 = RPs[o_feature_set].hierarchical(outer_models, outer_rmses, outer_errors,
                                                  i_feature_set,
                                                  i_regression_method,
                                                  i_regressor)
        # Store some of the RegressionProcessor attributes for later use
        x[(o_feature_set, i_feature_set)] = RPs[o_feature_set].x
        y[(o_feature_set, i_feature_set)] = RPs[o_feature_set].y
        p_docs[(o_feature_set, i_feature_set)] = RPs[o_feature_set].p_docs
        pp[(o_feature_set, i_feature_set)] = RPs[o_feature_set].pp
        block_list[(o_feature_set, i_feature_set)] = RPs[o_feature_set].block_list
        norm[(o_feature_set, i_feature_set)] = inner_norm
    norm[o_feature_set] = RPs[o_feature_set].norm
        
# Package the data that'll be used for plotting
DATA['GPinTPOT'] = {'models': models,
                    'rmses': rmses,
                    'errors': errors,
                    'x': x,
                    'y': y,
                    'p_docs': p_docs,
                    'blocks': blocks,
                    'block_list': block_list,
                    'pp': pp,
                    'norm': norm}


# ### Save

# In[ ]:

# Save the regressions
for o_feature_set in OUTER_FEATURE_SETS:
    for i_feature_set in INNER_FEATURE_SETS:
        # Save the models alone for GASpy_predict to use
        with open('pkls/models/GPinTPOT_model_'                   + i_feature_set + '-inside-' + o_feature_set + '_'                   + '-'.join(DATA['GPinTPOT']['blocks']) + '.pkl', 'wb') as f:
            pkl = {'model': DATA['GPinTPOT']['models'][(o_feature_set, i_feature_set)],
                   'pp': DATA['GPinTPOT']['pp'][(o_feature_set, i_feature_set)],
                   'norm': {'outer': DATA['GPinTPOT']['norm'][o_feature_set],
                            'inner': DATA['GPinTPOT']['norm'][(o_feature_set, i_feature_set)]}}
            pickle.dump(pkl, f)

        # Save the entire package to use later in this notebook
        data = {}
        for datum in ['models', 'rmses', 'errors', 'x', 'y', 'p_docs', 'block_list', 'pp']:
            data[datum] = DATA['GPinTPOT'][datum][(o_feature_set, i_feature_set)]
        with open('pkls/data/GPinTPOT_data_'                   + i_feature_set + '-inside-' + o_feature_set + '_'                   + '-'.join(DATA['GPinTPOT']['blocks']) + '.pkl', 'wb') as f:
            pickle.dump(data, f)

