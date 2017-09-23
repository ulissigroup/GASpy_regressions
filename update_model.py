
# coding: utf-8

# In[1]:

# Importing
import pdb
import sys
from regressor import GASpyRegressor
import gpickle
sys.path.insert(0, '../')
from gaspy.utils import vasp_settings_to_str

VASP_SETTINGS = vasp_settings_to_str({'gga': 'RP',
                                      'pp_version': '5.4',
                                      'encut': 350})


# In[2]:

from tpot import TPOTRegressor


# In[3]:

model_name = 'TPOT'
features = ['coordcount', 'ads']
responses = ['energy']
blocks = None


# In[4]:

tpot = TPOTRegressor(
                     generations=4,
                     population_size=16,
                     verbosity=2,
                     random_state=42,
                    )
TPOT = GASpyRegressor(features=features, responses=responses,
                      blocks=blocks, vasp_settings=VASP_SETTINGS)
TPOT.fit_tpot(tpot, model_name=model_name)


# In[9]:

gpickle.dump(TPOT)

