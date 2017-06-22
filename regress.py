
# coding: utf-8

# # regress.ipynb
# Author:  Kevin Tran <ktran@andrew.cmu.edu>
# 
# This python notebook performs regressions on data pulled from a local GASdb. It then saves these regressions into pickles (for later use) and creates parity plots of the regression fits.

# ## Importing

# In[1]:


from pprint import pprint   # for debugging
import sys
import math
import numpy as np
sys.path.append('..')
from vasp_settings_to_str import vasp_settings_to_str
from gas_pull import GASPullByMotifs as GasPull
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import alamopy
import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go


# ## Load data

# In[2]:


# Location of the *.db file
#DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/'  # Cori
DB_LOC = '/Users/KTran/Nerd/GASpy'                 # Local

# Calculation settings we want to look at
VASP_SETTINGS = vasp_settings_to_str({'gga': 'BF',
                                      'pp_version': '5.4.',
                                      'encut': 350})

# Pull the data from the Local database
GAS_PULL = GasPull(DB_LOC, VASP_SETTINGS, split=True)
X, Y, DATA, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, lb_ads, lb_coord = GAS_PULL.coordcount_ads_to_energy()


# ## Regressions

# ### SKLearn Linear Regression

# In[3]:


# Create a surrogate model using simple linear regressor
LR = LinearRegression()
LR.fit(X_TRAIN, Y_TRAIN)
LR.name = 'Linear'
pickle.dump({'model': LR,
             'pre_processors': {'coordination': lb_coord,
                                'adsorbate': lb_ads}},
            open('pkls/CoordcountAds_Energy_LR.pkl', 'w'))


# ### SKLearn Ensemble Regression

# In[4]:


# Create a surrogate model using SKLearn's gradient, boosting, ensemble method
GBE = GradientBoostingRegressor()
GBE.fit(X_TRAIN, Y_TRAIN)
GBE.name = 'GBE'
pickle.dump({'model': GBE,
             'pre_processors': {'coordination': lb_coord,
                                'adsorbate': lb_ads}},
            open('pkls/CoordcountAds_Energy_GBE.pkl', 'w'))


# ### Alamo Regression

# In[5]:


# Create a surrogate model using ALAMOpy
# Since Alamo can take awhile, we actually try to load a pickle of the previous run
# before calling alamopy. Simply delete the pickle if you want to re-run.
try:
    ALA = pickle.load(open('pkls/CoordcountAds_Energy_Ala.pkl', 'r'))['model']
except IOError:
    ALA = alamopy.doalamo(X_TRAIN, Y_TRAIN.reshape(len(Y_TRAIN), 1),
                          X_TEST, Y_TEST.reshape(len(Y_TEST), 1),
                          showalm=1,
                          linfcns=1,
                          expfcns=1,
                          logfcns=1,
                          monomialpower=(1, 2, 3),
                          multi2power=(1, 2, 3),
                          ratiopower=(1, 2, 3)
                     )
    ALA['name'] = 'Alamo'
    pickle.dump({'model': ALA,
                 'pre_processors': {'coordination': lb_coord,
                                    'adsorbate': lb_ads}},
                open('pkls/CoordcountAds_Energy_Ala.pkl', 'w'))
#pprint(ALA['model'])


# ## Plotting

# ### SKLearn

# In[6]:


# Create Plotly for each SKLearn model
for model in [LR, GBE]:
    traces = []
    # Create a parity plot where each adsorbate is shown. We do that by pulling out
    # data for each adsorbate and then plotting them.
    for ads in np.unique(DATA['adsorbate']):
        x = [X[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y = [Y[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y_predicted = model.predict(x)
        traces.append(go.Scatter(x=y_predicted, y=y, mode='markers', name=ads))
    # Create a diagonal line for the parity plot
    lims = [-4, 6]
    traces.append(go.Scatter(x=lims, y=lims,
                             line=dict(color=('black'), dash='dash'), name='Parity line'))
    # Format and plot
    layout = go.Layout(xaxis=dict(title='Regressed (eV)'),
                       yaxis=dict(title='DFT (eV)'),
                       title='Adsorption Energy as a function of (Coordination Count, Adsorbate); Model = %s; RMSE = %0.3f eV' \
                             % (model.name, math.sqrt(metrics.mean_squared_error(Y_TEST, model.predict(X_TEST)))))
    iplot(go.Figure(data=traces, layout=layout))


# ### Alamo

# In[7]:


# Create Pyplot plots for each dictionary-type model
for model in [ALA]:
    traces = []
    # Create a parity plot where each adsorbate is shown. We do that by pulling out
    # data for each adsorbate and then plotting them.
    for ads in np.unique(DATA['adsorbate']):
        x = [X[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y = [Y[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        # Do some footwork because Alamo returns a lambda function that doesn't accept np arrays
        def model_predict(factors):
            '''
            Turn a vector of input data, `factors`, into the model's guessed output. We use
            this function to do so because lambda functions suck. We should address this by
            making alamopy output a better lambda function.
            '''
            args = dict.fromkeys(range(0, len(factors)-1), None)
            for j, factor in enumerate(factors):
                args[j] = factor
            return model['f(model)'](args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25])
        y_predicted = map(model_predict, x)
        traces.append(go.Scatter(x=y_predicted, y=y, mode='markers', name=ads))
    # Create a diagonal line for the parity plot
    lims = [-4, 6]
    traces.append(go.Scatter(x=lims, y=lims,
                             line=dict(color=('black'), dash='dash'), name='Parity line'))
    # Format and plot
    layout = go.Layout(xaxis=dict(title='Regressed (eV)'),
                       yaxis=dict(title='DFT (eV)'),
                       title='Adsorption Energy as a function of (Coordination Count, Adsorbate); Model = %s; RMSE = %0.3f eV' \
                             % (model['name'], math.sqrt(metrics.mean_squared_error(Y_TEST, map(model_predict, X_TEST)))))
    iplot(go.Figure(data=traces, layout=layout))


# In[ ]:




