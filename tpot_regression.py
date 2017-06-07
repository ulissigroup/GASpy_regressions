import sys
sys.path.append('..')
from pprint import pprint
import numpy as np
from ase.db import connect
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tpot import TPOTRegressor
from vasp_settings_to_str import vasp_settings_to_str

# Calculation settings we want to look at
VASP_SETTINGS = vasp_settings_to_str({'gga': 'BF',
                                      'pp_version': '5.4.',
                                      'encut': 350})
# Define the factors and responses this model will look at. These strings should be callable keys
# of the ase-db rows.
FACTORS = ['coordination', 'adsorbate']
RESPONSES = ['energy']
# TPOT settings
GEN = 1
POP = 10
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
DB = connect('../adsorption_energy_database.db')
DATA['db_rows'] = [row for row in DB.select()
                   if all([row[key] == VASP_SETTINGS[key] for key in VASP_SETTINGS])]

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
for factor in FACTORS:
    X_TRAIN += (DATA['Train']['factors'][factor],)
    X_TEST += (DATA['Test']['factors'][factor],)
X_TRAIN = np.hstack(X_TRAIN)
X_TEST = np.hstack(X_TEST)
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

# Use TPOT to create a pipeline
tpot = TPOTRegressor(generations=GEN, population_size=POP, verbosity=2, random_state=RAN)
tpot.fit(X_TRAIN, Y_TRAIN)
print(tpot.score(X_TEST, Y_TEST))
tpot.export('tpot_pipeline_%s_%s_%s.py' % (GEN, POP, RAN))
