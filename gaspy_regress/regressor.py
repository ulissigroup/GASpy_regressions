'''
This class performs regressions and is then able to make predictions with the resulting
model. Note that you must specify general regressions settings and conditions for pulling
GASdb data to instantiate this class, and then call on any of the `fit_*` methods to
actually perform the regression on the features. Then you can use the `predict`
and `parity_plot` methods.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb  # noqa:  F401
import itertools
from collections import defaultdict
import copy
import warnings
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from .preprocessor import GASpyPreprocessor
from gaspy import utils, gasdb, defaults


class GASpyRegressor(object):
    '''
    All of the `fit_*` methods have similar output structures. Here it is:

    Resulting attributes:
        model_name  A string indicating the name of the method used. This is useful for
                    labeling the plots in the `parity_plot` method.
        residuals   A nested dictionary with the following structure:
                    residuals = {'block': {'split_data_set': resids}} where
                    `resids` is a np.array of model residuals
        _predict    A function that can turn turn a preprocessed input into a prediction.
                    Inputs:
                        inputs  A numpy array. 1st dimension shows different data points,
                                while the second dimension shows different numerical features
                        block   A tuple indicating the blocks of the model that you want to use
                    Output:
                        predictions A numpy array. 1st dimension shows different data points,
                                    while the second dimension shows different responses.
    '''
    def __init__(self, features, responses, blocks=None, dim_red=None,
                 fingerprints=None, vasp_settings=None, collection='adsorption',
                 energy_min='default', energy_max='default', f_max='default',
                 ads_move_max='default', bare_slab_move_max='default', slab_move_max='default',
                 date_min=None, date_max=None, train_size=1, dev_size=None, n_bins=20,
                 k_folds=None, random_state=42, time_series=False, **kwargs):
        '''
        Pull and preprocess the data that you want to regress. The "regression options"
        define how we want to perform the regression. "Pulling/filtering" options decide
        where we want to pull the data, what we want to pull, and what things we do not
        want to pull.

        Inputs (regression options):
            features    A list of strings for each of the features that you want
                        to include. These strings should correspond to the
                        1st-level hidden methods in `GASpyPreprocessor`, but
                        without the leading underscore.  For example:
                        features = ('coordcount', 'ads')
            responses   A list of strings for each of the responses that you
                        want to include.  Pretty much just like features.
            blocks      A list of strings for each of the fingerprints on which
                        the user wants to block
            dim_red     A string indicating the dimensionality reduction technique
                        you want to use. Defaults to `None`. Reference the
                        gaspy_regress.preprocessor module for more details.
            kwargs      Any arguments you want to pass to the dimesionality
                        reducer.

        Inputs (pulling/filtering options):
            fingerprints        Mongo queries of parameters that you want pulled.
                                Note that we automatically set some of these queries
                                based on the features and responses you are trying to use.
                                We also pull the bulk identity and the miller index
                                automatically, because... you should probably have those.
                                So you really only need to define mongo queries for any
                                extra information you want.
            vasp_settings       A string of vasp settings. Use the
                                vasp_settings_to_str function in GAspy
            collection          A string for the mongo db collection you want to pull from.
            date_min            Python datetime object that specifies the earliest date
                                that you'd like to pull data from
            date_max            Python datetime object that specifies the latest date
                                that you'd like to pull data from
            energy_min          The minimum adsorption energy to pull from the
                                adsorption DB (eV). If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters
            energy_max          The maximum adsorption energy to pull from the
                                adsorption DB (eV). If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters
            f_max               The upper limit on the maximum force on an atom
                                in the system. If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters
            ads_move_max        The maximum distance that an adsorbate atom may
                                move (angstrom). If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters
            bare_slab_move_max  The maxmimum distance that a slab atom may move
                                when it is relaxed without an adsorbate (angstrom).
                                If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters
            slab_move_max       The maximum distance that a slab atom may move
                                (angstrom). If 'default', then pulls the default
                                value from gaspy.defaults.doc_filters

        Inputs (data splitting options):
            train_size  A float between 0 and 1 indicating the fraction of your
                        data you want to allocate for training/development. If
                        it is set to 1, then we assume this is your "final model"
                        and that you have no test set. This should be done only
                        after tuning hyperparameters. Defaults to 1
            dev_size    A float between 0 and 1 indicating the fraction of the total
                        data that you want to allocate for development. Assuming that you
                        have a lot of data, we recommend a 70/20/10 split between training,
                        development, and test sets, respectively. So `train_size == 0.7`
                        and `dev_size == 0.2`. The test size is implicit. Note that
                        `dev_size` is ignored if `train_size == 1`. Defaults to `None`,
                        which yields do development set.
            k_folds     A positive integer >= 1 indicating how many k-folds you
                        want during cross-validation of the test set. 10 is a good
                        number if you are still tuning. Defaults to `None`, which skips
                        cross-validation.
            n_bins      A positive integer for how many bins you want to use to stratify the root
                        train/test split. This is ignored if train_size == 1. Defaults to 20.
                        If `None`, then no stratification is used and shuffling is turned off.
                        This is useful for time-series splitting.
            time_series If `True`, then it will split and sort based on date. `n_bins` will be
                        effectively ignored. In other words:  If `train_size == 0.5`, then
                        the regressor will train on the first half of the data that was generated.

        Resulting attributes:
            features        Same thing as the input. Used mainly for making file_name to save
            responses       Same thing as the input. Used mainly for making file_name to save
            blocks          Same thing as the input. Used mainly for making file_name to save
            x               A nested dictionary with the following structure:
                            x = {'block': {'split_data_set': np.array(INPUT_DATA)}}
            y               The same as `x`, but for the outputs, not the inputs
            docs            The same as `x`, but for the Mongo json documents
            pp              The instance of GASpyPreprocessor that was used to preprocess the
                            data set pulled by FeaturePuller. This is used to preprocess other data
                            to make future predictions.
            block_list      A list of tuples. Each tuple represents a different block, and
                            each element in these tuples corresponds to a different fingerprint
                            value. For example:  Say that `blocks` = ['adsorbate`, `mpid`] and
                            'adsorbate` could take values of ['H', 'O'], while 'mpid' could
                            take values of ['mp-23', 'mp-126']. Then `block_list` would be
                            [('H', 'mp-23'), ('O', 'mp-23'), ('H', 'mp-126'), ('O', 'mp-126')].
                            Note that the order of the values in the tuple corresponds to the
                            order in which the fingerprints are listed within `blocks`.
                            If there is no block, then `block_list` = (None,).
            indices_train   The indices that can be used to find the training set
                            from the entire set
            indices_test    The indices that can be used to find the test set
                            from the entire set
            n_jobs          This is actually set at a default of 1 and is not an argument.
                            Setting this to a higher value (after instantiation and before
                            prediction) enables parallel prediction for SKLearn-based models.
                            You should set this equal to the number of threads you are using.
        '''
        self.features = features
        self.responses = responses
        self.blocks = blocks
        # Set defaults for some of the filters
        default_filters = defaults.doc_filters()
        if energy_min == 'default':
            energy_min = default_filters['energy_min']
        if energy_max == 'default':
            energy_max = default_filters['energy_max']
        if f_max == 'default':
            f_max = default_filters['f_max']
        if ads_move_max == 'default':
            ads_move_max = default_filters['ads_move_max']
        if bare_slab_move_max == 'default':
            bare_slab_move_max = default_filters['bare_slab_move_max']
        if slab_move_max == 'default':
            slab_move_max = default_filters['slab_move_max']

        # Python doesn't like dictionaries being used as default values, so we initialize here
        if not vasp_settings:
            vasp_settings = utils.vasp_settings_to_str({})
        if not fingerprints:
            fingerprints = {}

        # Make sure that we are always pulling out/storing the mongo ID number
        fingerprints['mongo_id'] = '$_id'
        # Some features require specific fingerprints. Here, we make sure that those
        # fingerprints are included
        if 'coordcount' in features:
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
        if 'rnnc_count' in features:
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['nextnearestcoordination'] = '$processed_data.fp_final.nextnearestcoordination'
        if 'neighbors_coordcounts' in features:
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['neighborcoord'] = '$processed_data.fp_final.neighborcoord'
        if 'coordatoms_chemfp0' in features:
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['adsorbates'] = '$processed_data.calculation_info.adsorbate_names'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
        if 'neighbors_chemfp0' in features:
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['adsorbates'] = '$processed_data.calculation_info.adsorbate_names'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['neighborcoord'] = '$processed_data.fp_final.neighborcoord'
        if 'hash' in features:
            fingerprints['mpid'] = '$processed_data.calculation_info.mpid'
            fingerprints['miller'] = '$processed_data.calculation_info.miller'
            fingerprints['top'] = '$processed_data.calculation_info.top'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['nextnearestcoordination'] = '$processed_data.fp_final.nextnearestcoordination'  # noqa: E501
            fingerprints['neighborcoord'] = '$processed_data.fp_final.neighborcoord'
        # If we want to block by some fingerprint, then we had better pull it out.
        # Here are some common ones to make life easy.
        if blocks:
            if 'top' in blocks:
                fingerprints['top'] = '$processed_data.calculation_info.top'
            if 'coordination' in blocks:
                fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            if 'nextnearestcoordination' in blocks:
                fingerprints['nextnearestcoordination'] = '$processed_data.fp_final.nextnearestcoordination'  # noqa: E501
            if 'neighborcoord' in blocks:
                fingerprints['neighborcoord'] = '$processed_data.fp_final.neighborcoord'

        # Some responses require specific queries. Here, we make sure that the correct
        # queries are defined
        if 'energy' in responses:
            fingerprints['energy'] = '$results.energy'
        # And here we pull a couple of other things for good measure. Because we pretty
        # much always want these.
        fingerprints['mpid'] = '$processed_data.calculation_info.mpid'
        fingerprints['miller'] = '$processed_data.calculation_info.miller'
        fingerprints['adsorbates'] = '$processed_data.calculation_info.adsorbate_names'
        fingerprints['adslab_calculation_date'] = '$processed_data.FW_info.adslab_calculation_date'

        # Pull the data into a list of mongo (json) documents
        with gasdb.get_adsorption_client() as client:
            docs = gasdb.get_docs(client, collection, fingerprints,
                                  adsorbates=None,
                                  calc_settings=None,
                                  vasp_settings=vasp_settings,
                                  energy_min=energy_min,
                                  energy_max=energy_max,
                                  f_max=f_max,
                                  ads_move_max=ads_move_max,
                                  bare_slab_move_max=bare_slab_move_max,
                                  slab_move_max=slab_move_max)
        if not docs:
            raise Exception('Failed to find any data. Please check your query settings.')
        # Add the 'adsorbate' key to the dictionaries. Because why the hell not.
        for doc in docs:
            doc['adsorbate'] = doc['adsorbates'][0]

        # Sort the documents by time-series if the user wanted it
        if time_series:
            docs.sort(key=lambda doc: doc['adslab_calculation_date'])
            n_bins = None
        # Or parse out docs that are too old/new
        if date_min:
            docs = [doc for doc in docs if date_min <= doc['adslab_calculation_date']]
        if date_max:
            docs = [doc for doc in docs if doc['adslab_calculation_date'] <= date_max]

        # Preprocess the features
        pp = GASpyPreprocessor(docs, features, dim_red=dim_red, **kwargs)
        x = pp.transform(docs)
        # Pull out, stack (if necessary), and numpy-array-ify the responses.
        # We might do real preprocessing to these one day. But not today.
        if len(responses) == 1:
            y = np.array([doc[responses[0]] for doc in docs])
        elif len(responses) > 1:
            y = []
            for response in responses:
                y.append(np.array([doc[response] for doc in docs]))
            y = np.concatenate(tuple(y), axis=1)

        # If we're training on everything, then assign the data to the class attributes and move on
        self.pp = pp
        if train_size == 1:
            self.x = {(None,): {'train': x,
                                'all': x}}
            self.y = {(None,): {'train': y,
                                'all': y}}
            self.docs = {(None,): {'train': docs,
                                   'all': docs}}
        # If we're splitting, then start splitting
        else:
            # Do a stratified split
            if n_bins:
                y_train, y_test, x_train, x_test, docs_train, docs_test = \
                    self._stratified_split(n_bins, train_size, random_state, y, x, docs)
            # If there are no bins (and therefore no stratification), then do a normal
            # train test split without any shuffling
            else:
                y_train, y_test, x_train, x_test, docs_train, docs_test = \
                    train_test_split(y, x, docs, train_size=train_size,
                                     random_state=random_state, shuffle=False)
            # Now store the information in class attributes
            self.x = {(None,): {'test': x_test,
                                'all': x}}
            self.y = {(None,): {'test': y_test,
                                'all': y}}
            self.docs = {(None,): {'test': docs_test,
                                   'all': docs}}
            # Do it all again, but for the development set. Note that we re-calculate
            # `dev_size` because we're splitting from the training set, not the whole set.
            if dev_size:
                dev_size = dev_size/train_size
                y_train, y_dev, x_train, x_dev, docs_train, docs_dev = \
                    self._stratified_split(n_bins, 1-dev_size, random_state,
                                           y_train, x_train, docs_train)
                self.x[(None,)]['dev'] = x_dev
                self.y[(None,)]['dev'] = y_dev
                self.docs[(None,)]['dev'] = docs_dev
            self.x[(None,)]['train'] = x_train
            self.y[(None,)]['train'] = y_train
            self.docs[(None,)]['train'] = docs_train

        # TODO:  Finish this K-folding
        # Now do the k-folding on the training set
        if k_folds:
            # Unpack the data we'll be folding
            x = self.x[(None,)]['train']
            y = self.y[(None,)]['train']
            docs = self.docs[(None,)]['train']

        if blocks:
            # TODO:  Address this when we start doing co-adsorption.
            # If we're blocking by adsorbate, then we create a new fingerprint, `adsorbate`,
            # from the fingerprint `adsorbates`. Note that the latter is a list of adsorbates,
            # while the former is simply the first adsorbate. This really only works
            # because we're only looking at one adsorbate at a time right now.
            if 'adsorbate' in blocks:
                for dataset in self.docs[(None,)]:
                    for i, doc in enumerate(self.docs[(None,)][dataset]):
                        self.docs[(None,)][dataset][i]['adsorbate'] = doc['adsorbates'][0]

            # Warn the user if they're trying to block by something that they might not
            # be pulling
            for block in blocks:
                if block not in self.docs[(None,)][dataset][0]:
                    warnings.warn('You are trying to block by %s, but we did not find that fingerprint'
                                  % block, SyntaxWarning)

            # `block_values` is a list of sublists, where each sublist contains all of the unique
            # values for each of the fingerprints specified in `blocks`. The order of the sublists
            # corresponds to the order of the fingerprints in the `blocks` list. For example,
            # if blocks = ['adsorbate', 'top'], then block_values could be
            # [['O', 'CO'], ['Top', 'Bottom']]. We use block_values to create `block_list`.
            unique_blocks = []
            docs = self.docs[(None,)]['all']
            for block in blocks:
                block_values = [doc[block] for doc in docs]
                unique_values = np.unique(block_values).tolist()
                unique_blocks.append(unique_values)
            self.block_list = [block for block in itertools.product(*unique_blocks)]

            # Filter the class attributes for each block, and then add the filtered
            # data to the attributes as sub-dictionaries
            for block in self.block_list:
                filtered_docs = {}
                filtered_x = {}
                filtered_y = {}
                for dataset, docs in self.docs[(None,)].iteritems():
                    filtered_docs[dataset] = []
                    filtered_x[dataset] = []
                    filtered_y[dataset] = []
                    for doc, x, y in zip(docs, self.x[(None,)][dataset], self.y[(None,)][dataset]):
                        doc_block = tuple([doc[b] for b in blocks])
                        if doc_block == block:
                            filtered_docs[dataset].append(doc)
                            filtered_x[dataset].append(x)
                            filtered_y[dataset].append(y)
                    filtered_x[dataset] = np.array(filtered_x[dataset])
                    filtered_y[dataset] = np.array(filtered_y[dataset])
                self.docs[block] = filtered_docs
                self.x[block] = filtered_x
                self.y[block] = filtered_y

        # If there is no blocking, then set `block_list` to [(None,)], which will cause this
        # class' methods to act on the entire dataset
        else:
            self.block_list = [(None,)]


    def _stratified_split(self, n_bins=20, train_size=0.80, random_state=42, *arrays):
        '''
        Perform a stratified train/test split. Note that you can pass any number of arrays to split,
        but the stratification is done one the first array that you pass to this method.

        Inputs:
            n_bins          A positive integer indicating the number of bins to use during
                            stratification. Higher values lead to better stratification,
                            but may also lead to too few data points per bin.
            train_size      A float between 0 and 1 that indicates the training size you want
                            to split by.
            random_state    Any number that you can feed to SKLearn's `train_test_split`
            *arrays         Arrays that you can pass to SKLearn's `train_test_split`. You need
                            at least one array.
        Outputs:
            split_data      A tuple of information created by SKLearn's `train_test_split`
        '''
        # Pull out the array that we'll be stratifying on and call it `y`
        y = arrays[0]

        # We make bins and put our predictions into these bins. This gives us `y_binned`,
        # which SK's `train_test_split` can then use to do a stratified train/test split.
        bins = np.linspace(0, len(y), n_bins)
        y_binned = np.digitize(y, bins)

        # We have an EAFP wrapper to make sure that the user specifies
        # a small enough number of bins.
        try:
            split_data = train_test_split(*arrays, train_size=train_size,
                                          stratify=y_binned, random_state=random_state)
        except ValueError as err:
            import sys
            raise type(err), type(err)(err.message +
                                       '\nTry decreasing n_bins when initializing GASpy_Regressor'), sys.exc_info()[2]

        return split_data


    def fit_sk(self, regressor, x_dict=None, y_dict=None, blocks=None, model_name=None):
        '''
        This method will assume that the regressor object you are passing it is an SKLearn
        type object, and will thus have `fit` and `predict` methods.

        Inputs:
            regressor   An SKLearn-type regressor object (e.g., GaussianProcessRegressor)
            x_dict      The same thing as `self.x`, but the user may specify something
                        to use instead of `self.x`.
            y_dict      The same thing as `self.y`, but the user may specify something
                        to use instead of `self.y`.
            blocks      A list of tuples indicating the blocks that you want to perform
                        the regression on.
            model_name  If you want to name this model something differently, then go
                        ahead. Doing so might reduce regressor saving conflicts.
        '''
        # Set defaults
        if not x_dict:
            x_dict = self.x
        if not y_dict:
            y_dict = self.y
        if not blocks:
            blocks = self.block_list
        if not model_name:
            model_name = 'sk'
        self.model_name = model_name

        # Initialize the outputs
        models = dict.fromkeys(blocks)
        residuals = dict.fromkeys(blocks)

        for block in blocks:
            # Initialize some more structuring for the outputs
            residuals[block] = dict.fromkeys(x_dict[block])
            # Copy the regressor template and perform the regression
            models[block] = copy.deepcopy(regressor)
            models[block].fit(x_dict[block]['train'], y_dict[block]['train'])

            # Post-process the results for each set of training, testing, and
            # all
            for dataset, y in y_dict[block].iteritems():
                y_hat = models[block].predict(x_dict[block][dataset])
                residuals[block][dataset] = y - y_hat

        # Create the model
        def _predict(docs, block=(None,), layer='outer'):
            '''
            Note that we assume that we are passing only one item to this function.
            We do this because we assume the `self` method will take care of
            the parallelization of multiple inputs.
            '''
            # Make sure we use the correct preprocessor
            if layer == 'outer':
                features = self.pp.transform(docs)
            elif layer == 'inner':
                features = self.pp_inner.transform(docs)
            model = models[block]
            predictions = model.predict(features)
            return predictions

        # Assign the attributes
        self._predict = _predict
        self.residuals = residuals
        self.models = models


    def fit_tpot(self, regressor, x_dict=None, y_dict=None, blocks=None, model_name=None):
        '''
        This method will assume that the regressor object you are passing it is a TPOT regressor
        type object, and will thus have `fit` and `predict` methods. And it will need
        to be modified in order for it to be pickled.

        Inputs:
            regressor   An TPOTRegressor object
            x_dict      The same thing as `self.x`, but the user may specify something
                        to use instead of `self.x`.
            y_dict      The same thing as `self.y`, but the user may specify something
                        to use instead of `self.y`.
            blocks      A list of tuples indicating the blocks that you want to perform
                        the regression on.
            model_name  If you want to name this model something differently, then go
                        ahead. Doing so might reduce regressor saving conflicts.
        '''
        # Set defaults
        if not x_dict:
            x_dict = self.x
        if not y_dict:
            y_dict = self.y
        if not blocks:
            blocks = self.block_list
        if not model_name:
            model_name = 'tpot'
        self.model_name = model_name

        # Initialize the outputs
        models = dict.fromkeys(blocks)
        residuals = dict.fromkeys(blocks)

        for block in blocks:
            # Initialize some more structuring for the outputs
            residuals[block] = dict.fromkeys(x_dict[block])
            # Copy the regressor template and perform the regression
            models[block] = copy.deepcopy(regressor)
            models[block].fit(x_dict[block]['train'], y_dict[block]['train'])
            # Convert the TPOTRegressor into a pipeline, because we can't pickle
            # TPOTRegressors
            models[block] = models[block].fitted_pipeline_

            # Post-process the results for each set of training, testing, and
            # all
            for dataset, y in y_dict[block].iteritems():
                y_hat = models[block].predict(x_dict[block][dataset])
                residuals[block][dataset] = y - y_hat

        # Create the model
        def _predict(docs, block=(None,), layer='outer'):
            '''
            Note that we assume that we are passing only one item to this function.
            We do this because we assume the `self` method will take care of
            the parallelization of multiple inputs.
            '''
            # Make sure we use the correct preprocessor
            if layer == 'outer':
                features = self.pp.transform(docs)
            elif layer == 'inner':
                features = self.pp_inner.transform(docs)
            model = models[block]
            predictions = model.predict(features)
            return predictions

        # Assign the attributes
        self._predict = _predict
        self.residuals = residuals
        self.models = models


    def fit_hierarchical(self, outer_regressor, outer_method, outer_features,
                         blocks=None, model_name=None, dim_red=None, **kwargs):
        '''
        This method will wrap a regression model around the regression that you've already
        fit (using one of the other `fit_*` methods). In other words, it will try to fit
        a model on the residuals of your first regression.

        Note that you cannot nest this method within itself. Because I didn't feel like
        coding that.

        Inputs:
            blocks      A list of tuples indicating the blocks that you want to perform
                        the regression on.
            model_name  If you want to name this model something differently, then go
                        ahead. Doing so might reduce regressor saving conflicts.
            dim_red     A string indicating the dimensionality reduction technique
                        you want to use. Defaults to `None`. Reference the
                        gaspy_regress.preprocessor module for more details.
            kwargs      Any arguments you want to pass to the dimesionality
                        reducer.
        '''
        # Set defaults
        if not blocks:
            blocks = self.block_list
        if not model_name:
            model_name = 'hierarchical'
        self.model_name = model_name
        # Store the inner model's information into separate attributes before we overwrite them
        try:
            self.features_inner = copy.deepcopy(self.features)
            self.x_inner = copy.deepcopy(self.x)
            self.pp_inner = copy.deepcopy(self.pp)  # noqa: E501
            self._predict_inner = copy.deepcopy(self._predict)
            self.residuals_inner = copy.deepcopy(self.residuals)
            self.models_inner = copy.deepcopy(self.models)
        except AttributeError:
            raise AttributeError('You tried to fit an outer model without fitting an inner model')
        # Pull out docs (for ease of reading)
        docs = self.docs

        # Create the outer preprocessor
        try:
            pp = GASpyPreprocessor(docs[(None,)]['train'], outer_features,
                                   dim_red=dim_red, **kwargs)
        except KeyError:
            raise KeyError('You probably tried to ask for an outer feature, but did not specify an appropriate `fingerprints` query to pull the necessary information out.')
        # Preprocess docs again, but this time for the outer regressor
        x = copy.deepcopy(self.x_inner)
        for block in x:
            for dataset in x[block]:
                x[block][dataset] = pp.transform(docs[block][dataset])
        # Save the new attributes
        self.features = outer_features
        self.x = x
        self.pp = pp

        # Execute the outer regression
        getattr(self, outer_method)(outer_regressor, x_dict=x, y_dict=self.residuals_inner,
                                    blocks=blocks, model_name=model_name)
        # Appropriately store the new function that we just made
        self._predict_outer = copy.deepcopy(self._predict)

        # Create and save the hierarchical model
        def _predict(docs, block=(None,)):
            '''
            Note that we assume that we are passing only one item to this function.
            We do this because we assume the `self` method will take care of
            the parallelization of multiple inputs.
            '''
            inner_prediction = self._predict_inner(docs, block=block, layer='inner')
            outer_prediction = self._predict_outer(docs, block=block, layer='outer')
            prediction = inner_prediction + outer_prediction
            return prediction
        self._predict = _predict


    def predict(self, docs, block=(None,), processes=32, doc_chunk_size=1000):
        '''
        This method is a wrapper for whatever `_predict` function that we created with a `fit_*`
        method. The `_predict` function accepts preprocessed inputs. This method does
        the preprocessing for the user and passes it to _predict.

        Note that we do predictions in parallel. We also pass generators as argument inputs,
        which is the Pythonic "just in time" way to do it. Otherwise we might run into memory
        issues with the huge feature sets we're passing around.

        Inputs:
            docs            A list of Mongo-style json (dict) objects. Each item in the
                            list will be used to make one prediction (apiece).
            block           A tuple indicating the block of the model you want to use.
                            Defaults to (None,)
            processes       An integer for how many processors/threads you want to use
                            when doing predictions.
            doc_chunk_size  We use multiprocessing to do predictions. `chunks` dictates how many
                            predictions a child process should do before clearing its memory cache.
        Outputs:
            predictions     A flat numpy array of the predictions of each `doc` within `docs`
        '''
        # Turn the list of documents into chunks, i.e., an iterator that creates lists.
        # This is so we can multiprocess efficiently.
        def chunks(docs, size):
            iterator = iter(docs)
            for first in iterator:
                yield itertools.chain([first], itertools.islice(iterator, size-1))
        doc_chunk_iterator = chunks(docs, doc_chunk_size)
        # Calculate the number of chunks we have so we can pass that information
        # to our progress bar
        n_chunks = len(docs)/doc_chunk_size
        if not len(docs) % doc_chunk_size:
            n_chunks += 1

        # Make the predictions via multiprocessing
        print('Making predictions...')
        predictions = utils.map_method(self, '_predict', doc_chunk_iterator, chunked=True,
                                       block=block, processes=processes, n_calcs=n_chunks)
        return np.concatenate(predictions, axis=0).flatten()


    def calc_performance_metric(self, metric='rmse'):
        '''
        Calculate the performance metrics of the fitted model.

        Inputs:
            metric  A string indicating which performance metric you want plotted.
                    Can be 'rmse', 'mae', or 'mad' for "root mean squared error",
                    "mean absolute error", and "median absolute deviation", respectively.
        '''
        try:
            # Calculate and print the performance metrics for each block of data
            metric_values = defaultdict(dict)
            for block, _residuals in self.residuals.iteritems():
                for dataset, __residuals in _residuals.iteritems():
                    # Here are the different metrics we're able to calculate now
                    y = np.zeros(__residuals.shape)
                    y_hat = __residuals
                    if metric in set(['rmse', 'RMSE']):
                        metric_values[block][dataset] = np.sqrt(metrics.mean_squared_error(y, y_hat))
                    elif metric in set(['mae', 'MAE']):
                        metric_values[block][dataset] = metrics.mean_absolute_error(y, y_hat)
                    elif metric in set(['mad', 'MAD']):
                        metric_values[block][dataset] = metrics.median_absolute_error(y, y_hat)
                    else:
                        raise SyntaxError('"%s" is not a valid argument for "metric"' % metric)
            return metric_values

        # Tell the user what they probably did wrong
        except AttributeError as error:
            error.args = (error.args[0] + '; you probably just tried to print metrics before doing a fit')
            raise error


    def parity_plot(self, blocks=None, datasets=None, jupyter=True, plotter='plotly',
                    xlabel=None, ylabel=None, title=None, lims=None, shift=0.,
                    fname=None, figsize=None, s=None, alpha=0.4, font=None):
        '''
        Create a parity plot of the model that's been fit.

        Input:
            blocks      A list of the blocks that you want to plot. If `None`,
                        will show all blocks.
            datasets    A list of the datasets that you want to plot (e.g., 'train',
                        'dev', test', or 'all'). If `None` will show all datasets
                        excluding 'all'.
            jupyter     A boolean that you pass to tell this class whether you are
                        in a Jupyter notebook or not. This will change how it displays.
            plotter     A string indicating the plotting tool you want to use. It can
                        be 'plotly' or 'matplotlib'
            xlabel      A string for the x-axis label
            ylabel      A string for the y-axis label
            title       A string for the title name. If `default`,
                        it turns into a string made up of class attributes.
            lims        A list whose elements decide the bounds within
                        which to create the parity line.
            shift       A float indicating how far you want to shift the energy values.
                        This is useful for when you are adding entropic contributions
                        to zero point energies.
            figsize     A 2-tuple indicating the size of the panel. Defaults to (15, 15).
                        Only works when plotter == 'matplotlib'.
            fname       A string indicating the file name you want to save the figure as.
                        Only works when plotter == 'matplotlib'. Defaults to `None`, in which
                        case the figure will not be saved.
            s           An integer (or float?) indicating the size of the marker you
                        want to use. Only works when plotter == 'matplotlib'
            alpha       A float between 0 and 1 indicating the transparency you want
                        in the data. 0 is totally transparent and 1 is opaque.
                        Only works when plotter == 'matplotlib'.
            font        A dictionary that matplotlib accepts to establish the fonts you
                        want to use. Only works when plotter == 'matplotlib'
        Outputs:
            x       The values of the x-axis that you plot
            y       The values of the y-axis that you plot
            text    The values of the hovertext that you plot
        '''
        if jupyter and plotter == 'plotly':
            init_notebook_mode(connected=True)

        # Establish defaults
        if not title:
            title = 'Predicting %s using a[n] %s model' % (tuple(self.responses), self.model_name)
        if not figsize:
            figsize = (15, 15)
        if not lims:
            lims = [-4, 6]
        if not xlabel:
            xlabel = 'Simulated %s' % tuple(self.responses)
        if not ylabel:
            ylabel = 'Regressed %s' % tuple(self.responses)
        if not font:
            font = {'family': 'sans', 'style': 'normal', 'size': 20}
        if not self.x:
            raise RuntimeError('Trying to plot without performing a fit first.')
        if not blocks:
            blocks = self.residuals.keys()
        if not datasets:
            datasets = self.x[(None,)].keys()
            try:
                datasets.remove('all')
            except ValueError:
                pass

        # Initialize the outputs
        y_out = {}
        y_hat_out = {}
        text_out = {}

        # Set the figure size if we're using matplotlib
        if plotter == 'matplotlib':
            plt.figure(figsize=figsize)

        # Pick out the data for each block/dataset combination the user asked for
        for block in blocks:
            for dataset in datasets:
                y = self.y[block][dataset]
                docs = self.docs[block][dataset]
                residuals = self.residuals[block][dataset]
                # Calculate the model's prediction and enact any shifts
                y_hat = y - residuals
                y = y + shift
                y_hat = y_hat + shift

                # Plot it
                if plotter == 'plotly':
                    traces = []
                    # If we're using plotly, then add hovertext, `text`
                    text = ['']*len(docs)
                    for i, doc in enumerate(docs):
                        for key, value in doc.iteritems():
                            text[i] += '<br>' + str(key) + ':  ' + str(value)
                    traces.append(go.Scatter(x=y, y=y_hat,
                                             name=str((block, dataset)),
                                             mode='markers',
                                             text=text))
                    text_out[(block, dataset)] = text
                elif plotter == 'matplotlib':
                    plt.scatter(y, y_hat, s=s, alpha=alpha,
                                label='%s data of %s block' % (dataset, block))
                    sns.set_style('white')
                else:
                    raise Exception('"%s" is an unrecognized argument for "plotter"', plotter)
                # Add the information to the output to return
                y_out[(block, dataset)] = y
                y_hat_out[(block, dataset)] = y_hat

        # Make a parity line
        if plotter == 'plotly':
            traces.append(go.Scatter(x=lims, y=lims,
                                     name='parity line',
                                     line=dict(color=('black'), dash='dash')))
        elif plotter == 'matplotlib':
            plt.plot(lims, lims, 'k--')

        # Format and show the plot
        if plotter == 'plotly':
            layout = go.Layout(xaxis=dict(title=xlabel),
                               yaxis=dict(title=ylabel),
                               title=title)
            iplot(go.Figure(data=traces, layout=layout))
        elif plotter == 'matplotlib':
            matplotlib.rc('font', **font)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(lims)
            plt.ylim(lims)
            if fname:
                plt.savefig(fname, bbox_inches='tight', transparent=True)
            plt.legend()
            plt.show()

        # Return the data in case the user wants to do something with it
        return y_out, y_hat_out, text_out


    def residual_plot(self, figsize=None, xlim=None, xticks=None, xlabel='Residuals', font_scale=2):
        '''
        Create a panel of historgrams of the model residuals.

        Inputs:
            figsize     A 2-tuple indicating the size of the panel. Defaults to (15, 15)
            xlim        A sequence of floats indicating the x range you want to plot over.
            xticks      A list for the locations you want x-axis tick points at. Defaults to
                        letting matplotlib take care of it.
            xlabel      String indicating the x-axis label you want to put on the entire panel
            font_scale  Integer (or float?) indicating the scale of the font you want to use.
        '''
        # Set defaults
        if not figsize:
            figsize = (15, 15)
        if not xlabel:
            xlabel = 'Residuals'

        # Pull out the residuals
        try:
            residuals = self.residuals
        except AttributeError as error:
            error.args = (error.args[0] + '; you probably just tried to show residuals before doing a fit')
            raise error

        # Figure out the dimensions of the subplot, then create it
        blocks = residuals.keys()
        datasets = residuals.values()[0].keys()
        n_blocks = len(blocks)
        n_datasets = len(datasets)
        fig, axes = plt.subplots(nrows=n_datasets, ncols=n_blocks, figsize=figsize)

        # Create each of the subplots.
        for i, (block, _residuals) in enumerate(residuals.iteritems()):
            for j, (dataset, __residuals) in enumerate(_residuals.iteritems()):
                # Get the axis object for this subplot. Use EAFP to deal with
                # situations where we only have one row or one column
                try:
                    ax = axes[j, i]
                except IndexError:
                    ax = axes[max((i, j))]
                # Make the plot and format it
                if xticks:
                    ax.set_xticks(xticks)
                if xlim:
                    ax.set_xlim(xlim[0], xlim[1])
                ax.set_yticks([])
                sns.distplot(__residuals, kde=False, ax=ax)

        # Label the blocks and datasets if we have an array of panels
        if n_blocks > 1 and n_datasets > 1:
            for ax, block in zip(axes[0], residuals.keys()):
                ax.set_title(block)
            if n_datasets > 1:
                for ax, dataset in zip(axes[:, 0], residuals.values()[0].keys()):
                    ax.set_ylabel(dataset, size='large')
        # If we only have multiple datasets, then label accordingly
        elif n_datasets > 1:
            for ax, dataset in zip(axes, residuals.values()[0].keys()):
                ax.set_ylabel(dataset, size='large')
        # If we only have multiple blocks, then label accordingly
        elif n_blocks > 1:
            for ax, block in zip(axes, residuals.keys()):
                ax.set_title(block)

        # Label the x-axis for the entire subplot array
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(xlabel)

        # Show it
        sns.set(font_scale=font_scale)
        fig.tight_layout()
        plt.show()
