'''
Various functions that are designed to be called from cron.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy import gasdb
from gaspy.utils import read_rc
from gaspy_regress import fingerprinters
from gaspy_regress.utils import save_pipeline_predictions


def model_and_predict(adsorbate):
    '''
    Create a modeling pipeline, use it to make predictions
    on the catalog, and then save them.

    Arg:
        adsorbate   String for the adsorbate that you want to
                    make a model and predictions for.
    '''
    model_name = 'model0'

    # Fit the transformers and models
    docs = gasdb.get_adsorption_docs(adsorbates=[adsorbate])
    energies_dft = np.array([doc['energy'] for doc in docs])
    inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
    outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
    fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter, outer_fingerprinter)
    scaler = StandardScaler()
    pca = PCA()
    preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                       ('scaler', scaler),
                                       ('pca', pca)])
    features = preprocessing_pipeline.fit_transform(docs)
    tpot = TPOTRegressor(generations=1,
                         population_size=16,
                         offspring_size=16,
                         verbosity=2,
                         #random_state=42,
                         scoring='neg_median_absolute_error',
                         n_jobs=16)
    tpot.fit(features, energies_dft)

    # Make the pipeline
    steps = [('fingerprinter', fingerprinter),
             ('scaler', scaler),
             ('pca', pca)]
    for step in tpot.fitted_pipeline_.steps:
        steps.append(step)
    pipeline = Pipeline(steps)

    # Save the pipeline
    gasdb_location = read_rc('gasdb_path')
    file_name = gasdb_location + '/pipeline_%s_%s.pkl' % (adsorbate, model_name)
    with open(file_name, 'wb') as file_handle:
        pickle.dump(pipeline, file_handle)

    # Make and save predictions
    _ = save_pipeline_predictions(pipeline, adsorbate, model_name)  # noqa: F841
    model_and_predict(adsorbate)
