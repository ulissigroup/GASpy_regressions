'''
This module is really just a wrapper for Python's `dill` module, but with some naming automation.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions (for alamopy)


def dump_model(regressor, fname=None):
    '''
    Dump a file. If a file name is not supplied, then assume the file is a fitted
    GASpyRegressor and create a file name from its attributes.
    '''
    if not fname:
        # If it's a hierarchical model, then we need to save both feature sets
        try:
            fname = __concatenate_model_name(regressor.model_name,
                                  regressor.features_inner + regressor.features,
                                  regressor.responses,
                                  regressor.blocks)
        # If it's not hierarchical, then proceed as normal
        except AttributeError:
            fname = __concatenate_model_name(regressor.model_name,
                                  regressor.features,
                                  regressor.responses,
                                  regressor.blocks)

    with open(fname, 'wb') as f:
        pickle.dump(regressor, f)


def load_model(model_name, features, responses, blocks):
    '''
    Load a file given the attributes that you want from the regressor. Refer to
    the `GASpy_regressor` class in `regressor.py` for descriptions of this
    function's arguments (which are the attributes of that class).
    '''
    fname = __concatenate_model_name(model_name, features, responses, blocks)
    with open(fname, 'rb') as f:
        regressor = pickle.load(f)

    return regressor


def dump_predictions(data, fname=None, regressor=None, system=None):
    '''
    Dump a data set of predictions. If a file name is not supplied, the user
    must supply an instance of a fitted GASpyRegressor and a system. This
    function will then try to find the file from there.

    Inputs:
        data        Any object, really. We recommend using the output from one of the
                    `transform.*` functions.
        fname       A string for the file name you want to use to dump. If `None`,
                    then the user must supply `regressor` and `system` arguments to
                    have this function auto-name the file.
        regressor   A fitted instance of a GASpyRegressor
        system      A string indicating the system that these predictions are for.
                    This should correspond to the Excel file sheet name, like 'CO2RR'
                    or 'HER'
    '''
    if not fname:
        # If it's a hierarchical model, then we need to save both feature sets
        try:
            fname = __concatenate_prediction_name(regressor.model_name,
                                                  regressor.features_inner + regressor.features,
                                                  regressor.responses,
                                                  regressor.blocks,
                                                  system)
        # If it's not hierarchical, then proceed as normal
        except AttributeError:
            fname = __concatenate_prediction_name(regressor.model_name,
                                                  regressor.features,
                                                  regressor.responses,
                                                  regressor.blocks,
                                                  system)

    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_predictions(model_name, features, responses, blocks, system):
    '''
    Load a file given the attributes that you want from the regressor. Refer to
    the `GASpy_regressor` class in `regressor.py` for descriptions of this
    function's arguments (which are the attributes of that class).
    '''
    fname = __concatenate_prediction_name(model_name, features, responses, blocks, system)
    with open(fname, 'rb') as f:
        regressor = pickle.load(f)

    return regressor


def __concatenate_model_name(model_name, features, responses, blocks):
    '''
    This function turns a set of attributes into an appropriate and consistent file name
    for a regressed model. Refer to the `GASpyRegressor` class in `regressor.py` for
    desciptions of this function's arguments (which are the attributes of that class).
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


def __concatenate_prediction_name(model_name, features, responses, blocks, system):
    '''
    This function turns a set of attributes into an appropriate and consistent file name
    for a set of predictions. Refer to the `GASpyRegressor` class in `regressor.py` for
    desciptions of this function's arguments (which are the attributes of that class).
    '''
    # We're really just piggy-backing off of a different __concatenate function, but adding
    # a little more
    fname = __concatenate_model_name(model_name, features, responses, blocks)
    fname = 'pkls/' + system + '_predictions_' + fname[5:]

    return fname
