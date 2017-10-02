
# coding: utf-8

# In[ ]:

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


# In[ ]:

import copy
from tpot import TPOTRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


# In[ ]:

model_name = 'GP_around_TPOT'
features = ['coordcount', 'ads']
outer_features = ['neighbors_coordcounts']
responses = ['energy']
blocks = None
fingerprints = {'neighborcoord': '$processed_data.fp_final.neighborcoord'}


# In[ ]:

tpot = TPOTRegressor(
                     generations=4,
                     population_size=16,
                     verbosity=2,
                     random_state=42,
                    )
gp = GaussianProcessRegressor(
                              #kernel= 1.0*RBF(length_scale=0.05) \
                              #       +1.0*RBF(length_scale=0.2) \
                              #       +1.0*WhiteKernel(noise_level=0.05**2.0),
                              #n_restarts_optimizer=2,
                             )
H = GASpyRegressor(features=features, responses=responses,
                   blocks=blocks, vasp_settings=VASP_SETTINGS,
                   fingerprints=fingerprints)
H.fit_tpot(tpot, model_name=model_name)
H.fit_hierarchical(gp, 'fit_sk', outer_features, model_name=model_name)


# In[ ]:

gpickle.dump(H)

