'''
This script will take a data ball of predictions created by
`gaspy_regress.predict` and dump it to a Google Sheet.

Inputs:
    gsheet      A string indicating the name of the Google sheet you want to write to
    worksheet   A string indicating the name of the worksheet within the Google sheet you
                want to write to
Optional Inputs:
    predictions             A string indicating the location of the pickled form of the data ball.
                            If you don't pass this, then this script will pull a pickle given
                            various tags that are hard-coded within.
    comparison_predictions  A string indicating the location of the pickled form of the
                            secondary data ball. Doing this will add extra information
                            to the worksheet.
    identifying_labels      The labels of the data balls that we use to cross-reference
                            between the `predictions` data ball and the `comparison_predictions`
                            data ball. This must be passed in the format:
                                --identifying_labels 'foo, bar'
    reporting_labels        The labels of the `comparison_predictions` data ball that we want
                            to append to the data sheet. This must be passed in the format:
                                --reporting_labels 'foo, bar'
    comparison_name         A string indicating the prefix you want to attach to the
                            reporting labels
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb
import argparse
import pickle
import warnings
import gaspy_regress


# Unpack the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gsheet')
parser.add_argument('--worksheet')
parser.add_argument('--predictions')
parser.add_argument('--comparison_predictions')
parser.add_argument('--identifying_labels')
parser.add_argument('--reporting_labels')
parser.add_argument('--comparison_name')
args = parser.parse_args()
sheet = args.gsheet
worksheet = args.worksheet
prediction_path = args.predictions
comparisons_path = args.comparison_predictions
identifying_labels = args.identifying_labels.split(', ')
reporting_labels = args.reporting_labels.split(', ')
comparison_name = args.comparison_name

# If the user passed a path, then just open the data ball.
if prediction_path:
    with open(prediction_path, 'r') as f:
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
    data_ball = gaspy_regress.io.load_predictions(model_name, features+outer_features,
                                                  responses, blocks, system)
# Pass the data ball to the appropriate parsing/post-processing function
best_surfaces, labels = gaspy_regress.predict.best_surfaces(data_ball, performance_threshold=0.1)

# Now do the same thing for the data ball of comparison data
if comparisons_path:
    with open(comparisons_path, 'r') as f:
        comparison_ball = pickle.load(f)
    comparisons, _ = gaspy_regress.predict.best_surfaces(comparison_ball, performance_threshold=0.,
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
        except KeyError as err:
            message = 'We did not find the ' + str(identifier) + ' identifier in the data ball of comparison data; ignoring it'
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
gaspy_regress.io.push_to_gspread(best_surfaces, labels, sheet, worksheet)
