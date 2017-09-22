'''
This module is really just a wrapper for Python's `dill` module, but with some naming automation.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions (for alamopy)


def dump(regressor, fname=None):
    '''
    Dump a file. If a file name is not supplied, then assume the file is a fitted
    GASpyRegressor and create a file name from its attributes.
    '''
    if not fname:
        fname = __concatenate(regressor.model_name,
                              regressor.features,
                              regressor.responses,
                              regressor.blocks)

    with open(fname, 'wb') as f:
        pickle.dump(regressor, f)


def load(model_name, features, responses, blocks):
    '''
    Load a file given the attributes that you want from the regressor. Refer to
    the `GASpy_regressor` class in `regressor.py` for descriptions of this
    function's arguments (which are the attributes of that class).
    '''
    fname = __concatenate(model_name, features, responses, blocks)
    with open(fname, 'rb') as f:
        regressor = pickle.load(f)

    return regressor


def __concatenate(model_name, features, responses, blocks):
    '''
    This function turns a set of attributes into an appropriate and consistent file name.
    Refer to the `GASpyRegressor` class in `regressor.py` for desciptions of this
    function's arguments (which are the attributes of that class).
    '''
    # Turn the attributes into strings
    feature_names = 'FEATURES_' + '_'.join(features)
    response_names = 'RESPONSES_' + '_'.join(responses)
    try:
        block_names = 'BLOCKS_' + '_'.join(blocks)
    except TypeError:
        block_names = 'BLOCKS_'
    # Make the file name
    fname = 'pkls/' + '_'.join([model_name, feature_names, response_names, block_names]) + '.pkl'

    return fname
