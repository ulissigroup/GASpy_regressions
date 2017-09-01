'''
This class performs regressions using different feature sets and different regressions.
Each of the non-hidden methods are meant to be used the same way.

Inputs:     Changes depending on the method.
Outputs:
    models  A dictionary whose keys are the block ('no_block' if there is no blocking).
            The values are the model object. The type of model object varies, and it
            depends on the method called.
    rmses   A nested dictionary whose first set of keys are the block ('no_block' if there
            is no blocking). The second set of keys are the dataset (i.e., 'train',
            'test', or 'train+test'. The values of the sub-dictionary are the
            root-mean-squared-error of the model for the corresponding block
            and dataset.
    errors  The same as `rmses`, but it returns an np.array of errors instead of a
            float of RMSE values.
'''

import pdb
from pprint import pprint
import itertools
import copy
import sys
import math
import numpy as np
from sklearn import metrics
sys.path.append('..')
from pull_features import PullFeatures


class RegressionProcessor(object):
    def __init__(self, feature_set, blocks=None, **kwargs):
        '''
        This class is meant to be used to perform regressions on features that are pulled via
        `PullFeatures`.

        Inputs:
            feature_set A string of the feature set to perform the regressions on. This
                        string should be a method for `PullFeatures`.
            blocks      A list of strings for each of the features on which the user wants to block
            kwargs      Any arguments that the user may want to pass to `PullFeatures`,
                        such as `vasp_settings`.
        Resulting attributes:
            x           A dictionary of stacked arrays containing the data for all of the factors.
                        The keys are 'train', 'test', and 'train+test', and they correspond to
                        the training set, test/validation set, and the cumulation of the training/
                        test sets.
            y           The same as `x`, but for the outputs, not the inputs
            p_docs      The same as `x`, but the dict values are not np.arrays of data.
                        Instead, they are dictionaries with structures analogous to the
                        `p_docs` returned by `gaspy.utils.get_docs`, i.e., it is a dictionary
                        whose keys are the keys of `fingerprints` and whose values are lists
                        of the results.
            pp          A dict of the preprocessor used by `PullFeatures` to help turn the
                        fingerprints into the features.
            block_list  A list of tuples. Each tuple represents a different block, and
                        each element in these tuples corresponds to a different fingerprint
                        value. For example:  Say that `blocks` = ['adsorbate`, `mpid`] and
                        'adsorbate` could take values of ['H', 'O'], while 'mpid' could
                        take values of ['mp-23', 'mp-126']. Then `block_list` would be
                        [('H', 'mp-23'), ('O', 'mp-23'), ('H', 'mp-126'), ('O', 'mp-126')].
                        Note that the order of the values in the tuple corresponds to the
                        order in which the fingerprints are listed within `blocks`.
                        If there is no block, then `block_list` = 'no_block'.
        '''
        # Initialize the outputs
        self.x = {}
        self.y = {}
        self.p_docs = {}
        # Pull out the features, and then assign them to the class attributes
        puller = PullFeatures(**kwargs)
        self.x['no_block'], self.y['no_block'], self.p_docs['no_block'], self.pp = \
                getattr(puller, feature_set)()

        if blocks:
            # `block_values` is a list of sublists, where each sublist contains all of the unique
            # values for each of the fingerprints specified in `blocks`. The order of the sublists
            # corresponds to the order of the fingerprints in the `blocks` list. We use it to
            # create `block_list`
            block_values = []
            for block in blocks:
                block_values.append(np.unique(self.p_docs['no_block']['train+test'][block]).tolist())
            self.block_list = [block for block in itertools.product(*block_values)]
            # Filter the class attributes for each block, and then add the filtered
            # data to the attributes as sub-dictionaries
            datasets = dict.fromkeys(self.x)    # e.g., ['train', 'test', 'train+test']
            for block in self.block_list:
                self.x[block] = self._filter(self.x['no_block'], blocks, block)
                self.y[block] = self._filter(self.y['no_block'], blocks, block)
                self.p_docs[block] = self._filter(self.p_docs['no_block'], blocks, block)

        # If there is no blocking, then set `block_list` to ['no_block'], which will cause this
        # class' methods to act on the entire dataset pulled by `PullFeatures`.
        else:
            self.block_list = ['no_block']


    def _filter(self, data, blocks, block):
        '''
        Filter the `data` according to the `block` that it belongs to.
        Note that the algorithm to create the `fdata` intermediary object is... complicated.
        I hurt my brain writing it. Feel free to pick it apart to make it easier to read.

        Inputs:
            data        A dictionary whose keys are 'train+test', 'train', and 'test'.
                        The values are numpy arrays of data that are yielded by `PullFeatures`...
                        or they are dictionaries of parsed mongo data are also yielded by
                        `PullFeatures`
            blocks      A list of the names of the fingerprints that we are blocking on,
                        e.g., ['adsorbate', 'mpid']
            block       A tuple of the values of the fingerprints values that we are blocking on,
                        e.g., ('H', 'mp-126'). The order of the block values must
                        match the order of block names in the `block_names` list.
        Output:
            filtered_data   The subset of `data` whose fingerprint values match those supplied
                            in `block`
        '''
        # Initialize output
        filtered_data = dict.fromkeys(data)
        # Find the type of the values of `data` so that we treat it correctly
        dtype = type(data.values()[0])

        # If `_data` is an np.array, then treat it as such. This probably means
        # that `_data` is either `x` or `y`
        if dtype == type(np.array([])):
            for dataset, _data in data.iteritems():
                fdata = [datum for i, datum in enumerate(_data)
                         if all([fp_value == self.p_docs['no_block'][dataset][blocks[j]][i]
                                 for j, fp_value in enumerate(block)])]
                # Convert to np.array so that it can be accepted by most regressors
                filtered_data[dataset] = np.array(fdata)

        # If `_data` is a dict, then we need to loop through each element. This
        # probably means that `_data` is `p_docs`.
        elif dtype == dict:
            for dataset, _data in data.iteritems():
                filtered_data[dataset] = dict.fromkeys(_data)
                for p_doc_key, __data in _data.iteritems():
                    fdata = [datum for i, datum in enumerate(__data)
                             if all([fp_value == self.p_docs['no_block'][dataset][blocks[j]][i]
                                     for j, fp_value in enumerate(block)])]
                    filtered_data[dataset][p_doc_key] = fdata

        return filtered_data


    def sk_regressor(self, model):
        '''
        This method will assume that the model object you are passing it is an SKLearn
        type object, and will thus have `fit` and `predict` methods.

        Output:
            models  The values within `models` will be the same types as the input
                    `model`. So if the user supplies a GaussianProcessRegressor type
                    model, then that's what comes out.
        '''
        # Initialize the outputs
        models = dict.fromkeys(self.block_list)
        rmses = dict.fromkeys(self.block_list)
        errors = dict.fromkeys(self.block_list)

        for block in self.block_list:
            # Initialize some more structuring for the outputs
            rmses[block] = dict.fromkeys(self.x[block])
            errors[block] = dict.fromkeys(self.x[block])
            # Copy the model template and perform the regression
            models[block] = copy.deepcopy(model)
            models[block].fit(self.x[block]['train'], self.y[block]['train'])

            # Post-process the results for each set of training, testing, and
            # train+test data
            for dataset in self.x[block].keys():
                y = self.y[block][dataset]
                y_hat = models[block].predict(self.x[block][dataset])
                mse = metrics.mean_squared_error(y, y_hat)
                rmses[block][dataset] = math.sqrt(mse)
                errors[block][dataset] = y_hat - y

        return models, rmses, errors


    def tpot(self, model):
        '''
        This method will assume that the model object you are passing it is a TPOT model
        type object, and will thus have `fit` and `predict` methods. And it will need
        to be modified in order for it to be pickled.

        Output:
            models      It will be a TPOTRegressor.fitted_pipeline_ object
        '''
        # It turns out TPOT works exactly as SKLearn does!
        models, rmses, errors = self.sk_regressor(model)

        # All we need to do is to pull out the fitted pipelines from the standard
        # TPOT objects
        for block in self.block_list:
            models[block] = models[block].fitted_pipeline_

        return models, rmses, errors


    # TODO:  Finish writing this part. Namely, figure out/remember how to get the lambda function
    # out of the alamopy output so that we can apply it to the other data sets
    def alamo(self, **kwargs):
        '''
        Use alamopy to perform the regression.

        Inputs:
            kwargs  The same key word arguments that you would pass to alamopy.doalamo,
                    excluding the training and testing data.
        Output:
            models  foo
        '''
        # Initialize the outputs
        models = dict.fromkeys(self.block_list)
        rmses = dict.fromkeys(self.block_list)
        errors = dict.fromkeys(self.block_list)

        for block in self.block_list:
            # Initialize some more structuring for the outputs
            rmses[block] = dict.fromkeys(self.x[block])
            errors[block] = dict.fromkeys(self.x[block])
            # Perform the regression
            models[block] = alamopy.doalamo(self.x[block]['train'], self.y[block]['train'],
                                            self.x[block]['test'], self.y[block]['test'],
                                            **kwargs)

            # Post-process the results for each set of training, testing, and
            # train+test data
            for dataset in self.x[block].keys():
                y = self.y[block][dataset]
                y_hat = 'foo'
                mse = metrics.mean_squared_error(y, y_hat)
                rmses[block][dataset] = math.sqrt(mse)
                errors[block][dataset] = y_hat - y

        return models, rmses, errors


    # TODO:  Finish this. The inner and outer modeling is done, but we still need to package them
    def hierarchical_regression(self, outer_method, outer_model, inner_method, inner_model):
        '''
        This method accepts the results of many of the other methods of this class and
        then tries to fit another model to regress the subsequent erros of the original
        model.

        Inputs:
            outer_method    The `Regress` method to be used to create the outer model
            outer_model     The `model` object that should be used by the outer model
            inner_method    The `Regress` method to be used to create the inner model
            inner_model     The `model` object that should be used by the inner model
        Outputs:
            Identical to the standard outputs of the other methods in this class, but
            all have additional keys:  'outer_model' and 'inner_model'. The values in
            these keys are the standard outputs of the outer_model and inner model
            (respectively).
        '''
        # Initialize the outputs
        models = {}
        rmses = {}
        errors = {}
        models['inner_model'] = dict.fromkeys(self.block_list)
        rmses['inner_model'] = dict.fromkeys(self.block_list)
        error['inner_model'] = dict.fromkeys(self.block_list)
        # Call the outer method and store the results to the outputs
        outer_model, outer_rmses, outer_errors = outer_method(outer_model)
        models['outer_model'] = outer_model
        rmses['outer_model'] = outer_rmses
        errors['outer_model'] = outer_errors

        for block in self.block_list:
            # Initialize some more structuring for the outputs
            rmses['inner_model'][block] = dict.fromkeys(self.x[block])
            errors['inner_model'][block] = dict.fromkeys(self.x[block])
            # Copy the model template and perform the regression
            models['inner_model'][block] = copy.deepcopy(model)
            models['inner_model'][block].fit(self.x[block]['train'], outer_errors[block]['train'])

            # Post-process the results for each set of training, testing, and
            # train+test data
            for dataset in self.x[block].keys():
                y = outer_errors[block][dataset]
                y_hat = models['inner_model'][block].predict(self.x[block][dataset])
                mse = metrics.mean_squared_error(y, y_hat)
                rmses['inner_model'][block][dataset] = math.sqrt(mse)
                errors['inner_model'][block][dataset] = y_hat - y

        return models, rmses, errors
