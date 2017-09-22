'''
This class contains methods to convert mongo documents to and from preprocessed
vectors that are meant to be fed into regressors.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb
import copy
from collections import OrderedDict
import numpy as np
from sklearn import preprocessing


class GASpyPreprocessor(object):
    '''
    We transform fingerprints into features by creating preprocessing functions
    for each of the features, and then concatenating the outputs of each of these functions
    into a single numpy array. The preprocessing functions are created in the hidden
    methods (i.e., ones that start with `_`).
    '''
    # pylint: disable=missing-docstring
    def __init__(self, p_docs, features):
        '''
        This `__init__` method creates the list of preprocessing functions, `preprocessors`.
        It also creates the feature scaler, `scaler`. In other words:  This class is ready
        to start preprocessing right after it's instantiated. No need to do a `.fit`.

        Input:
            p_docs      The parsed mongo docs that you want to pre-process into features.
                        Note that these parsed mongo docs are used to create the preprocessing
                        algorithms. The parsed mongo docs passed to the `transform` method
                        will be transformed using the algoritms created by `__init__`.
            features    A list of strings, where each string corresponds to the
                        feature that you want to be created. The available strings are the
                        hidden methods in this class, but without the leading underscores
                        (e.g., 'coordcount'). Order matters, because the pre-processed
                        features will be stacked in order of appearance in this list.
        Resulting attributes:
            p_docs          The same as the `p_docs` input, but simply saved as an attribute
            preprocessors   An ordered dictionary whose keys are features and whose
                            values are the sklearn.preprocessing classes for each feature
            scaler          An instance of the sklearn.preprocessing.StandardScaler class
                            that has been fit to the preprocessed, stacked `p_docs`
        '''
        self.p_docs = p_docs

        # Create the preprocessors for each feature
        self.preprocessors = OrderedDict.fromkeys(features)
        for feature in features:
            self.preprocessors[feature] = getattr(self, '_'+feature)()

        # [Partially] preprocess the fingerprints, then stack them together
        numerical_features = [preprocessor(p_docs) for preprocessor in self.preprocessors.values()]
        if len(numerical_features) > 1:
            stacked_features = []
            for i in range(len(numerical_features[0])):
                datum = [numerical_feature[i] for numerical_feature in numerical_features]
                stacked_features.append(np.concatenate(tuple(datum)))
            stacked_features = np.array(stacked_features)
        elif len(numerical_features) == 1:
            stacked_features = numerical_features[0]
        # Create and fit the scaler
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit_transform(stacked_features)


    def _energy(self, p_docs=None):
        '''
        Create a preprocessing function for adsorption energy. Yes, this is pretty silly.
        But it's CONSISTENT!

        Input:
            p_docs  Parsed mongo dictionaries. Default value of `None` yields `self.p_docs`
        Output:
            preprocess_energy   Function to turn floats into a numpy array
        '''
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)

        def preprocess_energy(p_docs):
            return np.array(p_docs)

        return preprocess_energy


    def __create_symbol_lb(self, p_docs=None):
        '''
        Create a label binarizer that can convert a symbol string into a binary vector

        Inputs:
            p_docs  Parsed mongo dictionaries over which to perform the fitting.
                    Default value of `None` yields `self.p_docs`
        Outputs:
            lb          The label binarizer used to turn the string into an array of binary counts
            coordcount  The actual vector of integers that are the preprocessed form of `p_docs`
        '''
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)

        # Pull out all of the symbols that we can find in `p_docs`
        symbols = [symbols for symbols in p_docs['symbols']
                   # TODO:  Remove this conditional after we start looking at carbides
                   # or oxides. We only put this here to appease ALAMO
                   if (symbols != 'C' and symbols != 'O')]

        # Find all of the unique symbols. Then fit a label binarizer on them
        # that will be able to convert a single string into a binary vector.
        unq_symbols = np.unique([item for sublist in symbols for item in sublist])
        lb = preprocessing.LabelBinarizer()
        lb.fit(unq_symbols)

        return lb


    def _coordcount(self, p_docs=None, return_lb=False):
        '''
        Create a preprocessing function for coordination count

        Input:
            p_docs      Parsed mongo dictionaries. Default value of `None` yields `self.p_docs`
            return_lb   If `True`, then return the binarizer. Used mainly by the `_rnnc_cou
        Output:
            preprocess_coordcount   Function to preprocess parsed mongo docs into
                                    an array of binary vectors
        '''
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)

        # Create a label binarizer to convert a single symbol string into a vector of binaries
        lb = self.__create_symbol_lb(p_docs)

        def preprocess_coordcount(p_docs):
            '''
            Transform mongo documents into a vector of integers.
            We do this by using the binarizer to turn a coordination into a
            list of sparse binary vectors. For example: We turn `Ag-Au-Au`
            into [[1 0 0 0 0], [0 1 0 0 0], [0 1 0 0 0]]. We then use np.sum
            to sum each list into a coordcount, [1 2 0 0 0].
            '''
            # Unpack
            coords = p_docs['coordination']
            # Calculate
            coordcounts = np.array([np.sum(lb.transform(coord.split('-')), axis=0)
                                    for coord in coords])
            return coordcounts

        return preprocess_coordcount


    def _rnnc_count(self, p_docs=None):
        '''
        Create a preprocessing function for reduced, next-nearest-neighbor coordination count.
        The "reduced" part means that we subtract out coordcount from next-nearest-neighbor count.

        Input:
            p_docs  Parsed mongo dictionaries. Default value of `None` yields `self.p_docs`
        Output:
            preprocess_rnnc_count   Function to preprocess parsed mongo docs into
                                    an array of binary vectors
        '''
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)

        # Create a label binarizer to convert a single symbol string into a vector of binaries
        coordcount_transform, lb = self._coordcount(p_docs, return_lb=True)

        def preprocess_rnnc_count(p_docs):
            '''
            We do the same thing that we do in the `_coordcount` method, but for
            nextnearestcoordination instead. And then we substract out `coordcount`.
            '''
            # Unpack
            nncs = p_docs['nextnearestcoordination']
            # Calculate
            nnc_counts = np.array([np.sum(lb.transform(coord.split('-')), axis=0)
                                   for nnc in nncs])
            rnnc_counts = nnc_counts - coordcount_transform(p_docs)
            return rnnc_counts

        return preprocess_rnnc_count


    def _ads(self, p_docs=None):
        '''
        Create a preprocessing function for adsorbate identity.

        Input:
            p_docs  Parsed mongo dictionaries. Default value of `None` yields `self.p_docs`
        Output:
            preprocess_ads  Function to preprocess parsed mongo docs into
                            an array of binary vectors
        '''
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)

        # The 'adsorbates' key returns adsorbates. We're only looking at one right now,
        # so we pull it out into a single-layered list (instead of a nested list).
        adsorbates = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Find all of the unique adsorbate types. Then fit a label binarizer on them
        # that will be able to convert a single string into a binary vector.
        ads = np.unique(adsorbates)
        lb = preprocessing.LabelBinarizer()
        lb.fit(ads)

        # Create the preprocessing function
        def preprocess_ads(p_docs):
            ''' Transform mongo documents into an array of binary vectors '''
            ads_vector = np.array([lb.transform(ads)[0] for ads in p_docs['adsorbates']])
            return ads_vector

        return preprocess_ads


    # TODO:  Create this method
    def _gcn(self, p_docs=None):
        raise Exception('This method has not been created yet')


    # TODO:  Create this method
    def _dband_center(self, p_docs=None):
        raise Exception('This method has not been created yet')


    def _hash(self, p_docs=None, excluded_fingerprints='default'):
        '''
        Create a preprocessing function that hashes all of the inputs and then binarizes them.
        This is meant to create a unique identifier for each site, which is useful for testing.

        Input:
            p_docs                  Parsed mongo dictionaries. Default value of `None`
                                    yields `self.p_docs`
            excluded_fingerprints   A list of strings for all of the fingerprints inside of
                                    `p_docs` that the user does not want to hash. If `None`,
                                    then we set it as an empty list. If 'default', then set
                                    it to ['energy', 'adsorbates'].
        Output:
            preprocess_hash     Function to turn parsed mongo docs
        '''
        # Manage default arguments
        if not p_docs:
            p_docs = copy.deepcopy(self.p_docs)
        if not excluded_fingerprints:
            excluded_fingerprints = []
        elif excluded_fingerprints == 'default':
            excluded_fingerprints = ['energy']

        # Turn the list of excluded fingerprints into an empty dictionary so that we
        # can use the `in` statement later on
        excluded_fingerprints = dict.fromkeys(excluded_fingerprints)

        # Create a function to hash p_docs
        def __hash(p_docs):
            ''' Hash everything inside of `p_docs`, excluding the `excluded_fingerprints` '''
            # Number of data points
            n_data = len(p_docs.values()[0])

            # Initialize a list of strings. Each of these strings will be concatenated
            # strings of all of the fingerprints present in `p_docs`
            strs = ['']*n_data

            # Perform the concatenation, then hash all the entries
            for fingerprint, values in p_docs.iteritems():
                if fingerprint not in excluded_fingerprints:
                    for i in range(n_data):
                        strs[i] += str(values[i])
            hashes = [hash(_str) for _str in strs]

            return hashes

        # Create a label binarizer for all the unique hashes
        lb = preprocessing.LabelBinarizer()
        lb.fit(__hash(p_docs))

        # Lastly, create another function that wraps the binarizer around the hasher
        def preprocess_hash(p_docs):
            ''' Call on the `__hash` function, but now binarize its output '''
            _hashes = __hash(p_docs)
            bin_hashes = lb.transform(_hashes)
            return bin_hashes

        return preprocess_hash


    def transform(self, p_docs):
        '''
        Turn mongo documents into preprocessed, stacked, and scaled inputs.

        Input:
            p_docs  Parsed mongo documents, i.e., a dictionary of lists of data
        Output:
            preprocessed_features   A numpy array. The first axis extends for len(p_docs),
                                    and the second axis contains all of the preprocessed,
                                    scaled features.
        '''
        # Apply all of the preprocessing to the fingerprints, and store the results
        # in a list, `features`
        features = []
        for feature, preprocessor in self.preprocessors.iteritems():
            features.append(preprocessor(p_docs))

        # Stack the features and turn them into a numpy array
        preprocessed_features = np.concatenate(tuple(features), axis=1)

        return preprocessed_features


    # TODO:  Create this method
    def revert(self, processed_features):
        '''
        This method will turn a vector of processed features back into a mongo doc (dictionary)
        of information.

        Input:
            preprocessed_features   A numpy array. The first axis extends for len(p_docs),
                                    and the second axis contains all of the preprocessed,
                                    scaled features.
        Output:
            p_docs    A mongo-style list of dictionaries
        '''
        pass
