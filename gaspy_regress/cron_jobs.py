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
from gaspy.gasdb import get_catalog_docs_with_predictions
import datetime
import tqdm
from pymongo import UpdateOne
from gaspy.gasdb import get_mongo_collection


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


def ORR_limiting_potential_prediction(model_name):
    '''
    Take predictions for O/OH/OOH on each catalog site and
    calculate the limiting potential for the 4e ORR,
    adding it back to the catalog as predictions.orr_onset_potential_4e

    Arg:
        model_name  String for the model name from which to take
                    the predictions for O/OH/OOH.
    '''

    # Grab all of the catalog docs with predictions for O/OH/OOH
    docs = get_catalog_docs_with_predictions(['O', 'OH', 'OOH'])
    predictions = []
    for doc in docs:
        try:
            E_OH = next(iter(doc['predictions']['adsorption_energy']['OH'].values()))[1]
        except TypeError:
            # Sometime E_OH is not defined, not clear why
            E_OH = 5.

        G_O = next(iter(doc['predictions']['adsorption_energy']['O'].values()))[1] + 0.057
        G_OH = E_OH - 0.223
        G_OOH = next(iter(doc['predictions']['adsorption_energy']['OOH'].values()))[1] + 0.043

        # ORR limiting potential from Seoin Back, including free energy corrections
        predictions.append(np.min([4.92-G_OOH, G_OOH-G_O, G_O-G_OH, G_OH]))

    # Make a command for each doc to push the predictions
    mongo_commands = []
    time = datetime.datetime.utcnow()
    print('Making Mongo commands...')
    for doc, prediction in tqdm.tqdm(zip(docs, predictions), total=len(docs)):
        mongo_id = doc['mongo_id']
        energy_location = ('predictions.orr_onset_potential_4e.%s' % (model_name))
        command = UpdateOne({'_id': mongo_id},
                            {'$push': {energy_location: (time, prediction)},
                             '$set': {'mtime': time}})
        mongo_commands.append(command)

    # Write the ORR potential predictions
    with get_mongo_collection('relaxed_bulk_catalog') as collection:
        _ = collection.bulk_write(mongo_commands, ordered=False)    # noqa: F841
