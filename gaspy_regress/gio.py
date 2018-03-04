'''
This module is really just a wrapper for Python's `dill` module, but with some naming automation.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import pdb  # noqa: F401
import warnings
import dill as pickle
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from gaspy import utils
from . import predict


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
    cache_path = rc['gaspy_path'] + '/GASpy_regressions/cache/models/'

    # Turn the attributes into strings
    feature_names = 'FEATURES_' + '_'.join(features)
    response_names = 'RESPONSES_' + '_'.join(responses)
    try:
        block_names = 'BLOCKS_' + '_'.join(blocks)
    except TypeError:
        block_names = 'BLOCKS_'
    # Make the file name
    fname = cache_path + '_'.join([model_name, feature_names, response_names, block_names]) + '.pkl'

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
    cache_path, file_name = file_path.split('models/')
    fname = cache_path + 'predictions/' + system + '_predictions_' + file_name

    return fname


def gdrive_dump(gsheet, worksheet, predictions_path=None, comparisons_path=None,
                identifying_labels=None, reporting_labels=None, comparison_name=None):
    '''
    This script will take a data ball of predictions created by
    `gaspy_regress.predict` and dump it to a Google Sheet.

    Inputs:
        gsheet      A string indicating the name of the Google sheet you want to write to
        worksheet   A string indicating the name of the worksheet within the Google sheet you
                    want to write to
    Optional Inputs:
        predictions_path         A string indicating the location of the pickled form of the data ball.
                                If you don't pass this, then this script will pull a pickle given
                                various tags that are hard-coded within.
        comparisons_path        A string indicating the location of the pickled form of the
                                secondary data ball. Doing this will add extra information
                                to the worksheet.
        identifying_labels      The labels of the data balls that we use to cross-reference
                                between the `predictions` data ball and the `comparison_predictions`
                                data ball. This must be passed in a comma and space separated
                                string format, such as:  'MPID, Miller, Top?'
        reporting_labels        The labels of the `comparison_predictions` data ball that we want
                                to append to the data sheet. This must be passed in a comma and
                                space separated format, such as:  'dE [eV], Mongo ID'
        comparison_name         A string indicating the prefix you want to attach to the
                                reporting labels
    '''
    # If the user passed a path, then just open the data ball.
    if predictions_path:
        with open(predictions_path, 'r') as f:
            data_ball = pickle.load(f)
    # If the user did not pass a path, then define various parameter so that we can pull the data
    # via gaspy_regress's io
    else:
        model_name = 'GP_around_TPOT'
        features = ['coordcount']
        outer_features = ['neighbors_coordcounts']
        responses = ['energy']
        blocks = ['adsorbate']
        system = 'CO2RR'
        data_ball = load_predictions(model_name, features+outer_features, responses, blocks, system)
    # Pass the data ball to the appropriate parsing/post-processing function
    best_surfaces, labels = predict.best_surfaces(data_ball, performance_threshold=0.1)

    # Now do the same thing for the data ball of comparison data
    if comparisons_path:
        identifying_labels = identifying_labels.split(', ')
        reporting_labels = reporting_labels.split(', ')
        with open(comparisons_path, 'r') as f:
            comparison_ball = pickle.load(f)
        comparisons, _ = predict.best_surfaces(comparison_ball,
                                               performance_threshold=0.,
                                               max_surfaces=float('inf'))
        # Create a surface-property dictionary for the comparisons, `comparison_data`
        id_indices = [i for i, label in enumerate(labels) if label in identifying_labels]
        report_indices = [i for i, label in enumerate(labels) if label in reporting_labels]
        comparison_data = {}
        for surface in comparisons:
            identifier = tuple([surface[i] for i in id_indices])
            reported_values = tuple([surface[i] for i in report_indices])
            comparison_data[identifier] = reported_values
        # Now add the appropriate data to `best_surfaces`
        for s, surface in enumerate(best_surfaces):
            identifier = tuple([surface[i] for i in id_indices])
            try:
                reported_values = comparison_data[identifier]
            except KeyError:
                message = 'We did not find the ' + str(identifier) + \
                    ' identifier in the data ball of comparison data; ignoring it'
                warnings.warn(message, RuntimeWarning)
            surface = list(surface)
            surface.extend(reported_values)
            surface = tuple(surface)
            best_surfaces[s] = tuple(surface)
        # Fix the labeling
        reporting_labels = [comparison_name + ' ' + label for label in reporting_labels]
        labels = list(labels)
        labels.extend(reporting_labels)
        labels = tuple(labels)

    # Dump the data
    push_to_gspread(best_surfaces, labels, gsheet, worksheet)
