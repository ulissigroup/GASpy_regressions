'''
This module is really just a wrapper for Python's `dill` module, but with some naming automation.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import pdb
import dill as pickle
from oauth2client.service_account import ServiceAccountCredentials
from gaspy import utils
import gspread
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


def push_to_gspread(data, data_labels, sheet, worksheet):
    '''
    This function will rewrite a worksheet in a Google sheet using whatever data and labels
    you pass to it. Note that this function writes the data into rows instead of columns
    to improve writing speed. If you want column-organized data, then you can make a new
    worksheet and call the `transpose` function on the row-organized data.

    Inputs:
        data        A list of tuples containing the data you want to dump
        data_labels A tuple of strings containing the labels of the data. The length
                    of this tuple should be the same length as the tuples in `data`
        sheet       A string indicating the name of the Google sheet you want to be dumping to
        worksheet   A string indicating the name of the worksheet within the Google sheet you
                    want to dump to
    '''
    # Find and read the credentials so that we can access the spreadsheet
    gaspy_path = utils.read_rc()['gaspy_path']
    credentials_path = gaspy_path + '/GASpy_regressions/.gdrive_credentials.json'
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    gc = gspread.authorize(credentials)

    # Open a worksheet from a spreadsheet
    wks = gc.open(sheet)
    wk = wks.worksheet(worksheet)

    # Reorganize the data, label it, and then write it
    for i, label in enumerate(data_labels):
        data_vector = [datum[i] for datum in data]
        data_vector.insert(0, label)
        wk.insert_row(data_vector, index=i+1)

    # Clear the old data
    for _ in range(wk.row_count - len(data_labels)):
        wk.delete_row(len(data_labels)+1)


def __concatenate_model_name(model_name, features, responses, blocks):
    '''
    This function turns a set of attributes into an appropriate and consistent file name
    for a regressed model. Refer to the `GASpyRegressor` class in `regressor.py` for
    desciptions of this function's arguments (which are the attributes of that class).
    '''
    # Find the location of the repo to define the location of the pickle folder
    rc = utils.read_rc()
    pkl_path = rc['gaspy_path'] + '/GASpy_regressions/pkls/'

    # Turn the attributes into strings
    feature_names = 'FEATURES_' + '_'.join(features)
    response_names = 'RESPONSES_' + '_'.join(responses)
    try:
        block_names = 'BLOCKS_' + '_'.join(blocks)
    except TypeError:
        block_names = 'BLOCKS_'
    # Make the file name
    fname = pkl_path + '_'.join([model_name, feature_names, response_names, block_names]) + '.pkl'

    return fname


def __concatenate_prediction_name(model_name, features, responses, blocks, system):
    '''
    This function turns a set of attributes into an appropriate and consistent file name
    for a set of predictions. Refer to the `GASpyRegressor` class in `regressor.py` for
    desciptions of this function's arguments (which are the attributes of that class).
    '''
    # We're really just piggy-backing off of a different __concatenate function, but adding
    # a little more information to the front of the file name
    file_path = __concatenate_model_name(model_name, features, responses, blocks)
    pkl_path, file_name = file_path.split('pkls/')
    fname = pkl_path + 'pkls/' + system + '_predictions_' + file_name

    return fname
