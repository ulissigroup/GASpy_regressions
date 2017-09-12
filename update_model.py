
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

# ## SKLearn Gaussian Process

# ### Execute

# In[ ]:

# Specify the kernel to use. If it's `None`, then it uses SKLearn's default RBF
K = 1.0 * RBF(length_scale=0.05)+1.0*RBF(length_scale=0.2)+1.0*WhiteKernel(noise_level=0.05**2.0) 
#K = None
n_restarts = 2
# Create the model that you want to use to perform the regression
regressor = GaussianProcessRegressor(kernel=K, n_restarts_optimizer=n_restarts)
# Specify the model blocking. Use [] if you don't want blocking (this will help with saving)
#blocks = ['adsorbate']
blocks = []

# Initialize the results
models = dict.fromkeys(FEATURE_SETS)
rmses = dict.fromkeys(FEATURE_SETS)
errors = dict.fromkeys(FEATURE_SETS)
x = dict.fromkeys(FEATURE_SETS)
y = dict.fromkeys(FEATURE_SETS)
p_docs = dict.fromkeys(FEATURE_SETS)
block_list = dict.fromkeys(FEATURE_SETS)
pp = dict.fromkeys(FEATURE_SETS)
norm = dict.fromkeys(FEATURE_SETS)

for feature_set in FEATURE_SETS:
    # Pull the data out and store some of the processing information for plotting purposes
    rp = RegressionProcessor(feature_set, blocks=blocks, vasp_settings=VASP_SETTINGS)
    x[feature_set] = rp.x
    y[feature_set] = rp.y
    p_docs[feature_set] = rp.p_docs
    block_list[feature_set] = rp.block_list
    pp[feature_set] = rp.pp
    norm[feature_set] = rp.norm
    # Perform the regression
    models[feature_set], rmses[feature_set], errors[feature_set] =             rp.sk_regressor(regressor)

# Package the data that'll be used for plotting
DATA['GP'] = {'models': models,
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
for feature_set in FEATURE_SETS:
    # Save the models alone for GASpy_predict to use
    pkl = {'model': DATA['GP']['models'][feature_set],
           'pp': DATA['GP']['pp'][feature_set],
           'norm': DATA['GP']['norm'][feature_set]}
    with open('pkls/models/GP_model_' + feature_set + '_'               + '-'.join(DATA['GP']['blocks']) + '.pkl', 'wb') as f:
        pickle.dump(pkl, f)
        
    # Save the entire package to use later in this notebook
    data = {}
    for datum in ['models', 'rmses', 'errors', 'x', 'y', 'p_docs', 'block_list', 'pp']:
        data[datum] = DATA['GP'][datum][feature_set]
    with open('pkls/data/GP_data_' + feature_set + '_' +               '-'.join(DATA['GP']['blocks']) + '.pkl', 'wb') as f:
        pickle.dump(data, f)

