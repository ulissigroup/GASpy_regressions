'''
Various utility functions for the gaspy_regress package.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import datetime
import tqdm
from pymongo import UpdateOne
from gaspy.gasdb import get_mongo_collection, get_catalog_docs


def save_pipeline_predictions(pipeline, adsorbate, model_name):
    '''
    Given a pipeline created by GASpy_regressions, this function will
    use the pipeline to make predictions on our catalog of sites and
    then save those predictions to our Mongo database.

    Args:
        pipeline    A pipeline object that is fitted and is able to
                    turn the objects from `gaspy.gasdb.get_catalog_docs`
                    function into predictions via the `predict` method.
        adsorbate   [str] The adsorbate that you want to make predictions for.
        model_name  [str] The name of the pipeline/model you are using
    Returns:
        mongo_result    Mongo returns a `result` object after we write to it.
                        This is that object.
    '''
    # Use the pipeline to make predictions
    docs = get_catalog_docs()
    predictions = pipeline.predict(docs)

    # We'll be using pymongo's `bulk_write`, which takes
    # a list of commands. We'll be making a list of `UpdateOne` commands.
    mongo_commands = []
    time = datetime.datetime.utcnow()
    print('Making Mongo commands...')
    for doc, prediction in tqdm.tqdm(zip(docs, predictions), total=len(docs)):
        mongo_id = doc['mongo_id']
        energy_location = ('predictions.adsorption_energy.%s.%s'
                           % (adsorbate, model_name))
        command = UpdateOne({'_id': mongo_id},
                            {'$push': {energy_location: (time, prediction)},
                             '$set': {'mtime': time}})
        mongo_commands.append(command)

    # Write the results
    print('Writing predictions now...')
    with get_mongo_collection('relaxed_bulk_catalog') as collection:
        mongo_result = collection.bulk_write(mongo_commands, ordered=False)

    return mongo_result
