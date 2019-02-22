'''
Various functions that are designed to be called from cron.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from collections import defaultdict
from datetime import datetime
import pickle
import numpy as np
from pymongo import UpdateOne
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy.utils import read_rc, multimap_method
from gaspy.gasdb import get_catalog_docs, get_adsorption_docs, get_mongo_collection
from gaspy_regress import fingerprinters

GASDB_LOCATION = read_rc('gasdb_path')


def fit_model0_adsorption_energies(adsorbates=None):
    '''
    Create and save a modeling pipeline to predict adsortion energies.

    Arg:
        adsorbates  A list of strings indicating which adsorbate(s) to make
                    pipelines for
    Returns:
        pipeline    An `sklearn.pipeline.Pipeline` object that is fit to our
                    data and can be used to make predictions on adsorption
                    energies.  The pipeline is automatically saved to our GASdb
                    cache location, which is specified as 'gasdb_path' in the
                    `gaspyrc.json` file.
    '''
    model_name = 'model0'

    # Python doesn't like mutable default argumets
    if adsorbates is None:
        adsorbates = ['CO', 'H', 'O', 'OH', 'OOH', 'N']

    # Make a model for each adsorbate
    for adsorbate in adsorbates:
        print('[%s] Making %s pipeline/regression for %s...'
              % (datetime.utcnow(), model_name, adsorbate))

        # Fit the transformers and models
        docs = get_adsorption_docs(adsorbate=adsorbate)
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
        file_name = GASDB_LOCATION + '/pipeline_%s_%s.pkl' % (adsorbate, model_name)
        with open(file_name, 'wb') as file_handle:
            pickle.dump(pipeline, file_handle)


def save_predictions(models=None, adsorbates=None, processes=16):
    '''
    Wrapper to make a bunch of predictions, and then save them all at once. We
    save them all at once so that we can hopefully write to Mongo all at once
    instead of multiple times per day.

    Args:
        models      A list of strings indicating which model(s) to make
                    predictions for.
        adsorbates  A list of strings indicating which adsorbate(s) to make
                    predictions for
        processes   The number of threads/processes you want to be using
    Returns:
        mongo_result    Mongo returns a `result` object after we write to it.
                        This is that object.
    '''
    # Python doesn't like mutable default arguments
    if models is None:
        models = ['model0']
    if adsorbates is None:
        adsorbates = ['CO', 'H', 'N', 'O', 'OH', 'OOH']
    # Onset potentials need these adsorbates
    if 'O' not in adsorbates:
        adsorbates.append('O')
    if 'OH' not in adsorbates:
        adsorbates.append('OH')
    if 'OOH' not in adsorbates:
        adsorbates.append('OOH')

    # Get the adsorption energy predictions
    docs = get_catalog_docs()
    dE_predictions = _create_adsorption_energy_predictions(docs,
                                                           models=models,
                                                           adsorbates=adsorbates,
                                                           processes=processes)

    # Parse the predictions into `$push` commands
    adsorption_push_commands = __create_adsorption_energy_push_commands(docs, dE_predictions)
    orr_push_commands = __create_4e_orr_onset_potential_push_commands(docs, dE_predictions)

    # We'll be using pymongo's `bulk_write`, which takes a list of commands.
    # We'll be making a list of `UpdateOne` commands.
    mongo_commands = []
    for doc in docs:
        mongo_id = doc['mongo_id']
        push_commands = {**adsorption_push_commands[mongo_id],
                         **orr_push_commands[mongo_id]}
        command = UpdateOne({'_id': mongo_id},
                            {'$push': push_commands,
                             '$set': {'mtime': datetime.utcnow()}})
        mongo_commands.append(command)

    # Write the results
    print('[%s] Writing predictions into catalog now...' % datetime.utcnow())
    with get_mongo_collection('catalog') as collection:
        mongo_result = collection.bulk_write(mongo_commands, ordered=False)

    return mongo_result


def _create_adsorption_energy_predictions(docs, models, adsorbates, processes=16):
    '''
    Uses whatever pipeline we currently have saved, and then applies it to our
    catalog to predict adsorption energies.

    Args:
        docs        A list of dictionaries that can be fed to the modeling
                    pipelines to make predictions
        models      A list of strings indicating which model(s) to make
                    predictions for.
        adsorbates  A list of strings indicating which adsorbate(s) to make
                    predictions for
        processes   The number of threads/processes you want to be using
    Returns:
        all_predictions A dictionary whose keys are tuples of the model name
                        and adsorbate (respectively) and whose values are the
                        adsorption energy predictions of each document
    '''
    # Make predictions for each pair of models/adsorbates
    all_predictions = {}
    for model_name in models:
        for adsorbate in adsorbates:

            # Load the model/pipeline
            file_name = GASDB_LOCATION + '/pipeline_%s_%s.pkl' % (adsorbate, model_name)
            with open(file_name, 'rb') as file_handle:
                pipeline = pickle.load(file_handle)

            # Create the predictions
            print('[%s] Making adsorption energy predictions for %s using %s...'
                  % (datetime.utcnow(), adsorbate, model_name))
            predictions = multimap_method(pipeline, 'predict', docs, chunked=True,
                                          processes=processes, maxtasksperchild=100,
                                          chunksize=1000, n_calcs=len(docs))
            all_predictions[(model_name, adsorbate)] = predictions
    return all_predictions


def __create_adsorption_energy_push_commands(docs, all_predictions):
    '''
    Takes the predictions from `_create_adsorption_energy_predictions` and
    turns them into `$push` commands that Mongo can use to update our catalog.

    Args:
        docs            A list of dictionaries that can be fed to the modeling
                        pipelines to make predictions
        all_predictions The output of the
                        `_create_adsorption_energy_predictions` function.
    Returns:
        push_commands   A `defaultdict` whose keys are the Mongo IDs of each
                        adsorption site in the catalog, and whose values are
                        dictionaries that can be passed to the `$push` Mongo
                        command---e.g.,
                        `{'predictions.adsorption_energy.CO.model0': (datetime.datetime(*), -0.67),
                          'predictions.adsorption_energy.H.model0': (datetime.datetime(*), -0.07)}`
    '''
    push_commands = defaultdict(dict)
    for (model_name, adsorbate), predictions in all_predictions.items():
        prediction_location = ('predictions.adsorption_energy.%s.%s' % (adsorbate, model_name))
        for prediction, doc in zip(predictions, docs):
            mongo_id = doc['mongo_id']
            push_commands[mongo_id][prediction_location] = (datetime.utcnow(), prediction)
    return push_commands


def __create_4e_orr_onset_potential_push_commands(docs, all_predictions):
    '''
    Uses whatever pipeline we currently have saved, and then applies it to our
    catalog to predict ORR onset potentials.

    Args:
        docs            A list of dictionaries that can be fed to the modeling
                        pipelines to make predictions
        all_predictions The output of the
                        `_create_adsorption_energy_predictions` function.
    Returns:
        push_commands   A dictionary whose keys are the Mongo IDs of each
                        adsorption site in the catalog, and whose values are
                        dictionaries that can be passed to the `$push` Mongo
                        command---e.g.,
                        `{'predictions.orr_onset_potential_4e.model0': (datetime.datetime(*), -0.4)}`
    '''
    # Figure out what models we're using from the `all_predictions` dictionary
    models = set(model_name for model_name, _ in all_predictions.keys())

    # Fetch all of the adsorption energy predictions and convert them to dG
    push_commands = defaultdict(dict)
    for model_name in models:
        G_O = all_predictions[(model_name, 'O')] + 0.057
        G_OH = all_predictions[(model_name, 'OH')] - 0.223
        G_OOH = all_predictions[(model_name, 'OOH')] + 0.043

        # Calculate onset potential from dG (credit to Seoin Back)
        print('[%s] Making 4e onset potential predictions for using %s...'
              % (datetime.utcnow(), model_name))
        onset_potentials = np.min(np.array(4.92 - G_OOH,
                                           G_OOH - G_O,
                                           G_O - G_OH,
                                           G_OH),
                                  axis=0)

        # Parse the onset potentials into Mongo `$push` commands
        prediction_location = ('predictions.orr_onset_potential_4e.%s' % model_name)
        for doc, potential in zip(docs, onset_potentials):
            mongo_id = doc['mongo_id']
            push_commands[mongo_id][prediction_location] = (datetime.utcnow(), potential)
    return push_commands
