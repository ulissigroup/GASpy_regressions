import sys
sys.path.append('..')
from pprint import pprint
import numpy as np
from ase.db import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from vasp_settings_to_str import vasp_settings_to_str

# Calculation settings we want to look at
VASP_SETTINGS = vasp_settings_to_str({'gga': 'BF',
                                      'pp_version': '5.4.',
                                      'encut': 350})
# Define the factors and responses this model will look at. These strings should be callable keys
# of the ase-db rows.
FACTORS = ['coordination', 'adsorbate']
RESPONSES = ['energy']
# Location of the *.db file
DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/adsorption_energy_database.db'  # Cori
#DB_LOC = '../adsorption_energy_database.db'                                     # Local

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

# SKLearn requires us to pre-process our data. We do that here for each data set.
for key in FACTORS+RESPONSES:
    # Pull out the data
    data = [row[key] for row in DATA['db_rows']]

    # Specialized pre-processing for coordination, where we turn coordination into a vector
    # containing integers for the number of each element coordinated to the adsorbate.
    # Make sure this stays above the unicode elif
    if key == 'coordination':
        LB = preprocessing.LabelBinarizer()
        LB.fit(np.unique([item
                          for sublist in [row.symbols for row in DATA['db_rows']]
                          for item in sublist]))
        ppd_data = np.array(map(lambda coord: np.sum(LB.transform(coord.split('-')), axis=0), data))

    # Pre-process floats into np.arrays
    elif isinstance(data[0], float):
        ppd_data = np.array(data)

    # Pre-process integers
    elif isinstance(data[0], int):
        print('We have not yet decided how to deal with integers')

    # Pre-process unicode into categorical variables
    elif isinstance(data[0], unicode):
        cats, inv = np.unique(data,
                              return_inverse=True)
        # Binarize the categorical data
        enc = preprocessing.LabelBinarizer()
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
                                      test_size=0.25,
                                      random_state=42)
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
# Merge all of the responses into one array by combining the data into a tuple, which is then
# passed to a horizontal stacking function and converted to CSR sparse form.
# Do this for both the training and test set.
Y_TRAIN = tuple()
Y_TEST = tuple()
Y = tuple()
for response in RESPONSES:
    Y_TRAIN += (DATA['Train']['responses'][response],)
    Y_TEST += (DATA['Test']['responses'][response],)
Y_TRAIN = np.hstack(Y_TRAIN)
Y_TEST = np.hstack(Y_TEST)

# Use SKLearn's GradentBoostingRegressor to create a surrogate model
LR = LinearRegression()
LR.fit(X_TRAIN, Y_TRAIN)

# Print score and plot. This part of the code has not yet been tailored to
# handle multiple repsonses.
print(LR.score(X_TEST, Y_TEST))
PREDICT = LR.predict(X)
lims = [min(PREDICT+DATA['energy']), max(PREDICT+DATA['energy'])]
for ads_ind, ads in enumerate(DATA['Decoders']['adsorbate']):
    actual = []
    subset = []
    for data_ind, ads_list in enumerate(DATA['adsorbate'].tolist()):
        if ads_list.index(1) == ads_ind:
            subset.append(data_ind)
            actual.append(DATA['energy'][data_ind])
    predicted = LR.predict(X[subset])
    plt.scatter(predicted, actual, label=ads)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel('Predicted (eV)')
plt.ylabel('Actual (eV)')
plt.title('Linear Regression Fit for Adsorption Energy')
plt.legend()
plt.show()
plt.savefig('Fig_LR.pdf')
