
# coding: utf-8

# In[ ]:

# Importing
import pdb
from gaspy_regress import plot, predict, regressor
import gaspy_regress.io
from gaspy.utils import vasp_settings_to_str, read_rc

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
# blocks = None
fingerprints = {'neighborcoord': '$processed_data.fp_final.neighborcoord'}


# In[ ]:

H = gaspy_regress.io.load_model(model_name, features+outer_features, responses, blocks)


# In[ ]:

regressor = H
excel_file_path = read_rc()['gaspy_path'] + '/GASpy_regressions/volcanos_parsed.xlsx'


# In[ ]:

regressor_block = ('CO',)
adsorbate = 'CO'
system = 'CO2RR'
scale = 'log'


# In[ ]:

co2_data = predict.volcano(H, regressor_block, system, excel_file_path, scale, adsorbate)


# In[ ]:

gaspy_regress.io.dump_predictions(co2_data, regressor=H, system=system)


# In[ ]:

regressor_block = ('H',)
adsorbate = 'H'
system = 'HER'
scale = 'log'


# In[ ]:

her_data = predict.volcano(H, regressor_block, system, excel_file_path, scale, adsorbate)


# In[ ]:

gaspy_regress.io.dump_predictions(her_data, regressor=H, system=system)

