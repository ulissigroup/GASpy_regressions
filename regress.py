from pprint import pprint   # for debugging
import sys
sys.path.append('..')
import numpy as np
from vasp_settings_to_str import vasp_settings_to_str
from gas_pull import GASPullByMotifs as GasPull
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Location of the *.db file
#DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/'  # Cori
DB_LOC = '/Users/KTran/Nerd/GASpy'                 # Local
# Calculation settings we want to look at
VASP_SETTINGS = vasp_settings_to_str({'gga': 'BF',
                                      'pp_version': '5.4.',
                                      'encut': 350})

# Pull the data from the Local database
GP = GasPull(DB_LOC, VASP_SETTINGS, split=True)
X, Y, DATA, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = GP.coordcount_ads()

# Create a surrogate model using simple linear regressor
LR = LinearRegression()
LR.fit(X_TRAIN, Y_TRAIN)
LR.name = 'Linear'
# Create a surrogate model using SKLearn's gradient, boosting, ensemble method
GBE = GradientBoostingRegressor()
GBE.fit(X_TRAIN, Y_TRAIN)
GBE.name = 'GBE'

# Create a plot for each model
for model in [LR, GBE]:
    # Create a parity plot where each adsorbate is shown
    for ads in np.unique(DATA['adsorbate']):
        x = [X[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y = [Y[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y_predicted = model.predict(x)
        plt.scatter(y_predicted, y, label=ads)
    # `predict` is the model's prediction of what Y should be
    predict = model.predict(np.vstack((X_TRAIN, X_TEST)))
    # Create a diagonal line for the parity plot
    lims = [min(predict+DATA['energy']), max(predict+DATA['energy'])]
    plt.plot(lims, lims, '--k')
    # Label the plot and save it
    plt.xlabel('Regressed (eV)')
    plt.ylabel('DFT (eV)')
    plt.title('Adsorption Energy as a function of (Coordination Count, Adsorbate)\n\
              Model = %s\n\
              RMSE = %0.3f eV' % (model.name, metrics.mean_squared_error(Y, predict)))
    plt.legend()
    plt.savefig('CoordcountAds_%s.pdf' % model.name, bbox_inches='tight')
    plt.show()
