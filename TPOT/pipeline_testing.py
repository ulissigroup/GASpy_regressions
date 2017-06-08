import numpy as np
import sys
sys.path.append('/Users/KTran/Nerd/GASpy')      # Local
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
import matplotlib.pyplot as plt
from vasp_settings_to_str import vasp_settings_to_str
from ase.db import connect

# Calculation settings we want to look at
VASP_SETTINGS = vasp_settings_to_str({'gga': 'BF',
                                      'pp_version': '5.4.',
                                      'encut': 350})
# Define the factors and responses this model will look at. These strings should be callable keys
# of the ase-db rows.
FACTORS = ['coordination', 'adsorbate']
RESPONSES = ['energy']
# Location of the *.db file
#DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/adsorption_energy_database.db'  # Cori
DB_LOC = '/Users/KTran/Nerd/GASpy/adsorption_energy_database.db'                # Local
# TPOT settings
GEN = 100
POP = 100
RAN = 42

# Initialize a DATA dictionary. It will contain a key:value(s) pairing for each data set,
# regardless of whether the data set is a factor or response.
DATA = dict.fromkeys(FACTORS+RESPONSES, None)
# The DATA dictionary will also contain nested dictionaries for the training and testing
# subsets of data.
DATA['Train'] = {'factors': dict.fromkeys(FACTORS, None),
                 'responses': dict.fromkeys(RESPONSES, None)}
DATA['Test'] = {'factors': dict.fromkeys(FACTORS, None),
                'responses': dict.fromkeys(RESPONSES, None)}
# We will be encoding categorical (i.e., string) variables as ingeters using OneHotEncoder.
# To re-transform the encoded variables back to the categorical ones, we will store decoders
# in the DATA dictionary.
DATA['Decoders'] = dict.fromkeys(FACTORS+RESPONSES, None)

# Pull the data from the *.db and dump it into our DATA dictionary
DB = connect(DB_LOC)
DATA['db_rows'] = [row for row in DB.select()
                   if all([row[key] == VASP_SETTINGS[key] for key in VASP_SETTINGS])
                   and -4 < row.energy < 4
                   and row.max_surface_movement < 2]

# TPOT and SKLearn require us to pre-process our data. We do that here for each data set.
for key in FACTORS+RESPONSES:
    # Pull out the data
    data = [row[key] for row in DATA['db_rows']]

    # Pre-process floats into np.arrays
    if isinstance(data[0], float):
        ppd_data = np.array(data)

    # Pre-process integers
    elif isinstance(data[0], int):
        print('We have not yet decided how to deal with integers')

    # Pre-process unicode into categorical variables using OneHotEncoding
    elif isinstance(data[0], unicode):
        cats, inv = np.unique(data,
                              return_inverse=True)
        # Use the OneHotEncoder to preprocess the categorical data
        enc = LabelBinarizer()
        enc.fit(cats)
        ppd_data = enc.transform(data)
        # Store the `cats` object in the DATA dictionary for decoding the integers
        DATA['Decoders'][key] = cats

    # Pre-process booleans
    elif isinstance(data[0], bool):
        print('We have not yet decided how to deal with booleans')

    # Dump the pre-processed data into our dictionary
    DATA[key] = ppd_data

# Parse the data into training sets and test sets
for response in RESPONSES:
    for factor in FACTORS:
        split_data = train_test_split(DATA[factor],
                                      DATA[response],
                                      train_size=0.75,
                                      test_size=0.25)
        for i, data in enumerate(split_data):
            if i == 0:
                DATA['Train']['factors'][factor] = data
            elif i == 1:
                DATA['Test']['factors'][factor] = data
            elif i == 2:
                DATA['Train']['responses'][response] = data
            elif i == 3:
                DATA['Test']['responses'][response] = data

# Merge all of the factors into one array by combining the data into a tuple, which is then
# passed to a horizontal stacking function and converted to CSR sparse form.
# Do this for both the training and test set.
X_TRAIN = tuple()
X_TEST = tuple()
X = tuple()
for factor in FACTORS:
    X_TRAIN += (DATA['Train']['factors'][factor],)
    X_TEST += (DATA['Test']['factors'][factor],)
    X += (DATA[factor],)
X_TRAIN = np.hstack(X_TRAIN)
X_TEST = np.hstack(X_TEST)
X = np.hstack(X)
# Merge all of the factors into one array by combining the data into a tuple, which is then
# passed to a horizontal stacking function and converted to CSR sparse form.
# Do this for both the training and test set.
Y_TRAIN = tuple()
Y_TEST = tuple()
for response in RESPONSES:
    Y_TRAIN += (DATA['Train']['responses'][response],)
    Y_TEST += (DATA['Test']['responses'][response],)
Y_TRAIN = np.hstack(Y_TRAIN)
Y_TEST = np.hstack(Y_TEST)

# Use the pipeline created by TPOT
PL = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True,
                                                      max_features=0.65,
                                                      min_samples_leaf=19,
                                                      min_samples_split=9,
                                                      n_estimators=100)),
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, min_samples_split=14)
)
PL.fit(X_TRAIN, Y_TRAIN)
results = PL.predict(X_TEST)

# Print score and plot. This part of the code has not yet been tailored to
# handle multiple repsonses.
print(PL.score(X_TEST, Y_TEST))
PREDICT = PL.predict(X)
LIMS = [min(PREDICT+DATA['energy']), max(PREDICT+DATA['energy'])]
for ads_ind, ads in enumerate(DATA['Decoders']['adsorbate']):
    actual = []
    subset = []
    for data_ind, ads_list in enumerate(DATA['adsorbate'].tolist()):
        if ads_list.index(1) == ads_ind:
            subset.append(data_ind)
            actual.append(DATA['energy'][data_ind])
    predicted = PL.predict(X[subset])
    plt.scatter(predicted, actual, label=ads)
plt.xlim(LIMS)
plt.ylim(LIMS)
plt.xlabel('Predicted (eV)')
plt.ylabel('Actual (eV)')
plt.title('TPOT Fit for Adsorption Energy')
plt.legend()
plt.show()
plt.savefig('Fig_TPOT.pdf')
