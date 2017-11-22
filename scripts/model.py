
# coding: utf-8

# In[ ]:

# Importing
import pdb
from gaspy_regress.regressor import GASpyRegressor
from gaspy_regress import io, plot, predict
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
features = ['coordcount']
outer_features = ['neighbors_coordcounts']
responses = ['energy']
blocks = ['adsorbate']
fingerprints = {'neighborcoord': '$processed_data.fp_final.neighborcoord'}


# In[ ]:

tpot = TPOTRegressor(
                     generations=8,
                     population_size=32,
                     verbosity=1,
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
                   fingerprints=fingerprints, train_size=1)
H.fit_tpot(tpot, model_name=model_name)
H.fit_hierarchical(gp, 'fit_sk', outer_features, model_name=model_name)


# In[ ]:

io.dump_model(H)

