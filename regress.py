from pprint import pprint   # for debugging
import sys
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

# Location of the *.db file
DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/'  # Cori
#DB_LOC = '/Users/KTran/Nerd/GASpy'                 # Local
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
# Create a surrogate model using ALAMApy.
# Since Alamo can take awhile, we actually try to load a pickle of the previous run
# before calling alamopy. Simply delete the pickle if you want to re-run.
try:
    ALA = pickle.load(open('alamodel.pkl', 'r'))
except IOError:
    ALA = alamopy.doalamo(X_TRAIN, Y_TRAIN.reshape(len(Y_TRAIN), 1),
                          X_TEST, Y_TEST.reshape(len(Y_TEST), 1),
                          showalm=1,
                         )
    ALA['name'] = 'Alamo'
    pickle.dump(ALA, open('alamodel.pkl', 'w'), protocol=2)

# Create a plot for each dictionary-type model
#for model in []:
for model in [ALA]:
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
        plt.scatter(y_predicted, y, label=ads)
    # Create a diagonal line for the parity plot
    #predict = model.predict(X)
    #lims = [min(predict+DATA['energy']), max(predict+DATA['energy'])]
    lims = [-4, 6]
    plt.plot(lims, lims, '--k')
    # Label the plot and save it
    plt.xlabel('Regressed (eV)')
    plt.ylabel('DFT (eV)')
    plt.title('Adsorption Energy as a function of (Coordination Count, Adsorbate)\n\
              Model = %s\n\
              RMSE = %0.3f eV' \
              % (model['name'], metrics.mean_squared_error(Y_TEST, map(model_predict, X_TEST))))
    plt.legend()
    plt.savefig('CoordcountAds_%s.pdf' % model['name'], bbox_inches='tight')
    #plt.show()

sys.exit('Stop after Alamo is done')
# Create a plot for each SKLearn model
for model in [LR, GBE]:
    # Create a parity plot where each adsorbate is shown. We do that by pulling out
    # data for each adsorbate and then plotting them.
    for ads in np.unique(DATA['adsorbate']):
        x = [X[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y = [Y[i] for i, _ads in enumerate(DATA['adsorbate']) if _ads == ads]
        y_predicted = model.predict(x)
        plt.scatter(y_predicted, y, label=ads)
    # Create a diagonal line for the parity plot
    #predict = model.predict(X)
    #lims = [min(predict+DATA['energy']), max(predict+DATA['energy'])]
    lims = [-4, 6]
    plt.plot(lims, lims, '--k')
    # Label the plot and save it
    plt.xlabel('Regressed (eV)')
    plt.ylabel('DFT (eV)')
    plt.title('Adsorption Energy as a function of (Coordination Count, Adsorbate)\n\
              Model = %s\n\
              RMSE = %0.3f eV' \
              % (model.name, metrics.mean_squared_error(Y_TEST, model.predict(X_TEST))))
    plt.legend()
    plt.savefig('CoordcountAds_%s.pdf' % model.name, bbox_inches='tight')
    #plt.show()
