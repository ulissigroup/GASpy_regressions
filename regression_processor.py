'''
This class performs regressions using different feature sets and different regressions.
Each of the non-hidden methods are meant to be used the same way.

Inputs:     Changes depending on the method.
Outputs:
    regressors  A dictionary whose keys are the block ('no_block' if there is no blocking).
                The values are the model object. The type of model object varies, and it
                depends on the method called.
    rmses       A nested dictionary whose first set of keys are the block ('no_block' if there
                is no blocking). The second set of keys are the dataset (i.e., 'train',
                'test', or 'train+test'. The values of the sub-dictionary are the
                root-mean-squared-error of the model for the corresponding block
                and dataset.
    errors      The same as `rmses`, but it returns an np.array of errors instead of a
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
            x           A nested dictionary whose first set of keys are the blocks and whose
                        second seyt of keys are 'train', 'test', and 'train+test'. The values
                        are stacked arrays containing the data for all of the factors.
            y           The same as `x`, but for the outputs, not the inputs
            p_docs      The same as `x`, but the dict values are not np.arrays of data.
                        Instead, they are dictionaries with structures analogous to the
                        `p_docs` returned by `gaspy.utils.get_docs`, i.e., it is a dictionary
                        whose keys are the keys of `fingerprints` and whose values are lists
                        of the results.
            puller      The instance of the `PullFeatures` class used to... pull features.
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
            norm        A 1-D np.array containing the normalization factors for the features.
                        You need to (element-wise) divide new feature inputs by this vector
                        for the model to work correctly.
        '''
        # Initialize the outputs
        self.x = {}
        self.y = {}
        self.p_docs = {}
        self.blocks = blocks
        # Pull out the features, and then assign them to the class attributes
        self.puller = PullFeatures(**kwargs)
        self.x['no_block'], self.y['no_block'], self.p_docs['no_block'], self.pp, self.norm = \
                getattr(self.puller, feature_set)()

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
                fdata = [datum for i, datum in enumerate(_data) if all([fp_value == self.p_docs['no_block'][dataset][blocks[j]][i] for j, fp_value in enumerate(block)])]
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


    def sk_regressor(self, regressor, x_dict=None, y_dict=None):
        '''
        This method will assume that the regressor object you are passing it is an SKLearn
        type object, and will thus have `fit` and `predict` methods.

        Inputs:
            regressor   An SKLearn-type regressor object (e.g., GaussianProcessRegressor)
            x_dict      The same thing as `self.x`, but the user may specify something
                        to use instead of `self.x`.
            y_dict      The same thing as `self.y`, but the user may specify something
                        to use instead of `self.y`.
        Output:
            models  The values within `models` will be the same types as the input
                    `regressor`. So if the user supplies a GaussianProcessRegressor type
                    regressor, then that's what comes out.
        '''
        # Set defaults
        if not x_dict:
            x_dict = self.x
        if not y_dict:
            y_dict = self.y
        # Initialize the outputs
        models = dict.fromkeys(self.block_list)
        rmses = dict.fromkeys(self.block_list)
        errors = dict.fromkeys(self.block_list)

        for block in self.block_list:
            # Initialize some more structuring for the outputs
            rmses[block] = dict.fromkeys(x_dict[block])
            errors[block] = dict.fromkeys(x_dict[block])
            # Copy the regressor template and perform the regression
            models[block] = copy.deepcopy(regressor)
            models[block].fit(x_dict[block]['train'], y_dict[block]['train'])

            # Post-process the results for each set of training, testing, and
            # train+test data
            for dataset, y in y_dict[block].iteritems():
                y_hat = models[block].predict(x_dict[block][dataset])
                mse = metrics.mean_squared_error(y, y_hat)
                rmses[block][dataset] = math.sqrt(mse)
                errors[block][dataset] = y_hat - y

        return models, rmses, errors


    def tpot(self, regressor, x_dict=None, y_dict=None):
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
        Output:
            models      It will be a TPOTRegressor.fitted_pipeline_ object
        '''
        # Set defaults
        if not x_dict:
            x_dict = self.x
        if not y_dict:
            y_dict = self.y

        # It turns out TPOT works exactly as SKLearn does!
        models, rmses, errors = self.sk_regressor(regressor, x_dict=x_dict, y_dict=y_dict)
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
            rmses[block] = dict.fromkeys(x_dict[block])
            errors[block] = dict.fromkeys(x_dict[block])
            # Perform the regression
            models[block] = alamopy.doalamo(x_dict[block]['train'], y_dict[block]['train'],
                                            x_dict[block]['test'], y_dict[block]['test'],
                                            **kwargs)

            # Post-process the results for each set of training, testing, and
            # train+test data
            for dataset, y in y_dict[block].iteritems():
                y_hat = 'foo'
                mse = metrics.mean_squared_error(y, y_hat)
                rmses[block][dataset] = math.sqrt(mse)
                errors[block][dataset] = y_hat - y

        return models, rmses, errors


    def hierarchical(self, outer_models, outer_rmses, outer_errors,
                     inner_feature_set, inner_method, inner_regressor):
        '''
        This method accepts the results of many of the other methods of this class and
        then tries to fit another model to regress the subsequent erros of the original
        model. Note that this method assumes that you use the same blocking structure
        for both the inner and the outer methods, and that you also use the same GASpy_DB
        snapshot.

        Inputs:
            outer_models        The `models` for the outer model
            outer_rmses         The `rmses` for the outer model
            outer_errors        The `errors` for the outer model
            inner_feature_set   A string corresponding to the feature set for the inner model
            inner_method        The `Regress` method to be used to create the inner model
            inner_regressor     The regressing object that should be used by the inner model
        Outputs:
            models  A function that accepts the input to the outer model and the input
                    to the inner model to make a final prediction. All inputs should
                    probably be np.arrays. The outputs will probably np.arrays.
            rmses   This will be the same as "normal", but it will have two additional
                    keys:  'inner_model' and 'outer_model'. The subsequent values
                    will be identical to a normal `rmses` object, but specific to
                    either the inner or outer model.
            errors  Same as `rmses`, but for the errors instead
        '''
        # Initialize the outputs
        models = dict.fromkeys(self.block_list)
        rmses = dict.fromkeys(self.block_list)
        errors = dict.fromkeys(self.block_list)
        # Store the outer model information
        models['outer_model'] = outer_models
        rmses['outer_model'] = outer_rmses
        errors['outer_model'] = outer_errors

        # Initialize/pull the information for the inner feature set
        inner_x = {}
        inner_p_docs = {}
        inner_x['no_block'], _, inner_p_docs['no_block'], inner_pp, inner_norm \
                = getattr(self.puller, inner_feature_set)()
        # Filter the information for the inner feature set information
        if len(self.block_list) != 1:    # No need to filter if our only block is 'no_block'
            for block in self.block_list:
                inner_x[block] = self._filter(inner_x['no_block'], self.blocks, block)
                inner_p_docs[block] = self._filter(inner_p_docs['no_block'], self.blocks, block)
        # Add inner-feature-set information to the class attributes
        for block in self.block_list:
            for dataset, new_p_docs in inner_p_docs[block].iteritems():
                for fp_name, fp_value in new_p_docs.iteritems():
                    if fp_name not in self.p_docs[block][dataset]:
                        self.p_docs[block][dataset][fp_name] = fp_value
        for feature, _pp in inner_pp.iteritems():
            if feature not in self.pp:
                self.pp[feature] = _pp

        # Perform the inner model regression
        models['inner_model'], rmses['inner_model'], errors['inner_model'] = \
                getattr(self, inner_method)(inner_regressor,
                                            x_dict=inner_x,
                                            y_dict=errors['outer_model'])

        # Compile the outputs for the hierarchical model
        for block in self.block_list:
            # Initialize the sub-structure
            rmses[block] = dict.fromkeys(self.y[block])
            errors[block] = dict.fromkeys(self.y[block])
            # Calculate the rmses and the errors
            for dataset, y in self.y[block].iteritems():
                y_hat = y + errors['outer_model'][block][dataset] \
                        - models['inner_model'][block].predict(inner_x[block][dataset])
                mse = metrics.mean_squared_error(y, y_hat)
                rmses[block][dataset] = math.sqrt(mse)
                errors[block][dataset] = y_hat - y

            # Create a function that will serve as the hierarchical model
            def __h_model(x_outer, x_inner):
                '''
                Inputs:
                    x_outer An np.array that the outer model may accept directly in order
                            to make its prediction of the final solution
                    x_inner An np.array that the inner model may accept directly in order
                            to make its prediction of the outer model's error
                Outputs:
                    y_hat   An np.array that represents this hierarchical model's
                            final estimate of the solution
                '''
                # The outer model's estimate of the solution
                y_outer = models['outer_model'][block].predict(x_outer)
                # The inner model's estimate of the outer model's error
                y_inner = models['inner_model'][block].predict(x_inner)
                # The hierarchical model's estimate of the solution
                y_hat = y_outer - y_inner
                return y_hat
            models[block] = __h_model

        return models, rmses, errors, inner_x, inner_norm
