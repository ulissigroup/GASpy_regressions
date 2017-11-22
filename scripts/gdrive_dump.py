'''
This script will take a data ball of CO adsorption predictions created by
`gaspy_regress.predict.volcano` and dump it to a Google Sheet.

Inputs:
    [1] A string indicating the name of the Google sheet you want to write to
    [2] A string indicating the name of the worksheet within the Google sheet you
                want to write to
Optional input:
    [3] A string indicating the location of the pickled form of the data ball.
                    If you don't pass this, then this script will pull a pickle given
                    various tags that are hard-coded within.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import sys
import pickle
import gaspy_regress


# Pull out the arguments
sheet = sys.argv[1]
worksheet = sys.argv[2]

# If the user passed a path, then just open the data ball.
try:
    data_ball_path = sys.argv[3]
    with open(data_ball_path, 'r') as f:
        data_ball = pickle.load(f)
# If the user did not pass a path, then define various parameter so that we can pull the data
# via gaspy_regress's io
except IndexError:
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
# Dump the data
gaspy_regress.io.push_to_gspread(best_surfaces, labels, sheet, worksheet)
