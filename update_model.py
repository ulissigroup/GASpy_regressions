
# coding: utf-8

# In[1]:

# Importing
import pdb
import sys
from regressor import GASpyRegressor
import gpickle
import gplot
sys.path.insert(0, '../')
from gaspy.utils import vasp_settings_to_str

VASP_SETTINGS = vasp_settings_to_str({'gga': 'RP',
                                      'pp_version': '5.4',
                                      'encut': 350})


# In[2]:

import copy
from tpot import TPOTRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


# In[3]:

model_name = 'GP_around_TPOT'
features = ['coordcount']
outer_features = ['neighbors_coordcounts']
responses = ['energy']
blocks = ['adsorbate']
fingerprints = {'neighborcoord': '$processed_data.fp_final.neighborcoord'}


# In[7]:

tpot = TPOTRegressor(
                     generations=8,
                     population_size=32,
                     verbosity=1,
                    )
gp = GaussianProcessRegressor(
                              #kernel= 1.0*RBF(length_scale=0.05) \
                              #       +1.0*RBF(length_scale=0.2) \
                              #       +1.0*WhiteKernel(noise_level=0.05**2.0),
                              #n_restarts_optimizer=2,
                             )
H = GASpyRegressor(features=features, responses=responses,
                   blocks=blocks, vasp_settings=VASP_SETTINGS,
                   fingerprints=fingerprints, train_size=0.99)
H.fit_tpot(tpot, model_name=model_name)
H.fit_hierarchical(gp, 'fit_sk', outer_features, model_name=model_name)


# In[8]:

gpickle.dump(H)


# In[ ]:



