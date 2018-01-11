'''
This class contains methods to convert mongo documents to and from preprocessed
vectors that are meant to be fed into regressors.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb  # noqa:  F401
import copy
from collections import OrderedDict
import numpy as np
from sklearn import preprocessing
from gaspy import utils


class GASpyPreprocessor(object):
    '''
    We transform fingerprints into features by creating preprocessing functions
    for each of the features, and then concatenating the outputs of each of these functions
    into a single numpy array. The preprocessing functions are created in the hidden
    methods (i.e., ones that start with `_`).
    '''
    def __init__(self, docs, features):
        '''
        This `__init__` method creates the list of preprocessing functions, `preprocessors`.
        It also creates the feature scaler, `scaler`. In other words:  This class is ready
        to start preprocessing right after it's instantiated. No need to do a `.fit`.

        Input:
            docs        The  mongo docs that you want to pre-process into features.
                        Note that these mongo docs are used to create the preprocessing
                        algorithms. The mongo docs passed to the `transform` method
                        will be transformed using the algoritms created by `__init__`.
            features    A list of strings, where each string corresponds to the
                        feature that you want to be created. The available strings are the
                        hidden methods in this class, but without the leading underscores
                        (e.g., 'coordcount'). Order matters, because the pre-processed
                        features will be stacked in order of appearance in this list.
        Resulting attributes:
            docs            The same as the `docs` input, but simply saved as an attribute
            preprocessors   An ordered dictionary whose keys are features and whose
                            values are the sklearn.preprocessing classes for each feature
            scaler          An instance of the sklearn.preprocessing.StandardScaler class
                            that has been fit to the preprocessed, stacked `docs`
        '''
        self.docs = docs

        # Create the preprocessors for each feature
        self.preprocessors = OrderedDict.fromkeys(features)
        for feature in features:
            self.preprocessors[feature] = getattr(self, '_' + feature)()

        # Partially preprocess the fingerprints, then stack them into a numpy array
        numerical_features = [preprocess(docs) for preprocess in self.preprocessors.values()]
        stacked_features = np.concatenate(tuple(numerical_features), axis=1)

        # Create and fit the scaler
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(stacked_features)


    def _energy(self, docs=None):
        '''
        Create a preprocessing function for adsorption energy. Yes, this is pretty silly.
        But it's CONSISTENT!

        Input:
            docs  Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_energy   Function to turn floats into a numpy array
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        def preprocess_energy(docs):
            return np.array([doc['energy'] for doc in docs])

        return preprocess_energy


    def __create_symbol_lb(self, docs=None):
        '''
        Create a label binarizer that can convert a symbol string into a binary vector

        Inputs:
            docs    Mongo json dictionaries over which to perform the fitting.
                    Default value of `None` yields `self.docs`
        Outputs:
            lb          The label binarizer used to turn the string into an array of binary counts
            coordcount  The actual vector of integers that are the preprocessed form of `docs`
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Pull out all of the symbols that we can find in `docs`
        symbols = [doc['symbols'] for doc in docs
                   # TODO:  Remove this conditional after we start looking at carbides
                   # or oxides. We only put this here to appease ALAMO
                   if (doc['symbols'] != 'C' and doc['symbols'] != 'O')]

        # Find all of the unique symbols. Then fit a label binarizer on them
        # that will be able to convert a single string into a binary vector.
        unq_symbols = np.unique([item for sublist in symbols for item in sublist])
        lb = preprocessing.LabelBinarizer()
        lb.fit(unq_symbols)

        return lb


    def _coordcount(self, docs=None, return_lb=False):
        '''
        Create a preprocessing function for coordination count

        Input:
            docs        Mongo json dictionaries. Default value of `None` yields `self.docs`
            return_lb   If `True`, then return the binarizer. Used mainly by the `_rnnc_cou
        Output:
            preprocess_coordcount   Function to preprocess mongo docs into
                                    an array of binary vectors
            lb                      The label binarizer fitted to the symbols
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Create a label binarizer to convert a single symbol string into a vector of binaries
        lb = self.__create_symbol_lb(docs)

        def preprocess_coordcount(docs):
            '''
            Transform mongo documents into a vector of integers.
            We do this by using the binarizer to turn a coordination into a
            list of sparse binary vectors. For example: We turn `Ag-Au-Au`
            into [[1 0 0 0 0], [0 1 0 0 0], [0 1 0 0 0]]. We then use np.sum
            to sum each list into a coordcount, [1 2 0 0 0].
            '''
            # Package the calculation into a function so that we can parallelize it
            def _preprocess_coordcount(doc):  # noqa: E306
                coord = doc['coordination']
                coordcount = np.sum(lb.transform(coord.split('-')), axis=0)
                return coordcount
            # Use GASpy multiprocessing to do the calculation
            coordcounts = utils.multimap(_preprocess_coordcount, docs)
            return np.array(coordcounts)

        if not return_lb:
            return preprocess_coordcount
        else:
            return preprocess_coordcount, lb


    def _rnnc_count(self, docs=None):
        '''
        Create a preprocessing function for reduced, next-nearest-neighbor coordination count.
        The "reduced" part means that we subtract out coordcount from next-nearest-neighbor count.

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_rnnc_count   Function to preprocess mongo docs into
                                    an array of binary vectors
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Create a label binarizer to convert a single symbol string into a vector of binaries
        preprocess_coordcount, lb = self._coordcount(docs, return_lb=True)

        def preprocess_rnnc_count(docs):
            '''
            We do the same thing that we do in the `_coordcount` method, but for
            nextnearestcoordination instead. And then we substract out `coordcount`.
            '''
            # Package the calculation into a function so that we can parallelize it
            def _preprocess_rnnc_count(doc):  # noqa: E306
                ''' Yes, this is hacky. Please don't judge me. '''
                coord = doc['coordination']
                nnc = doc['nextnearestcoordination']
                nnc_count = np.sum(lb.transform(nnc.split('-')), axis=0)
                coordcount = preprocess_coordcount([{'coordination': coord}])
                rnnc_count = nnc_count - coordcount
                return list(rnnc_count)
            # Use GASpy multiprocessing to do the calculation
            rnnc_counts = utils.multimap(_preprocess_rnnc_count, docs)
            return np.array(rnnc_counts)

        return preprocess_rnnc_count


    def _ads(self, docs=None):
        '''
        Create a preprocessing function for adsorbate identity.

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_ads  Function to preprocess mongo docs into an array of binary vectors
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # The 'adsorbates' key returns adsorbates. We're only looking at one right now,
        # so we pull it out into a single-layered list (instead of a nested list).
        adsorbates = [adsorbates[0] for adsorbates in docs['adsorbates']]

        # Find all of the unique adsorbate types. Then fit a label binarizer on them
        # that will be able to convert a single string into a binary vector.
        ads = np.unique(adsorbates)
        lb = preprocessing.LabelBinarizer()
        lb.fit(ads)

        # Create the preprocessing function
        def preprocess_ads(docs):
            # We package the calculation into a function so that we can parallelize it
            def _preprocess_ads(doc):  # noqa: E306
                ads = doc['adsorbates']
                ads_vector = lb.transform(ads)[0]
                return ads_vector
            # Use GASpy multiprocessing to do the calculation
            ads_vectors = utils.multimap(_preprocess_ads, docs)
            return np.array(ads_vectors)

        return preprocess_ads


    def _neighbors_coordcounts(self, docs=None):
        '''
        Create a preprocessing function for the coordination counts of the adsorbate's neighebors.
        This is just like `_coordcount`, but will have a vector for each of the neighbors instead
        of one vector for the adsorbate itself.

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_rnnc_count   Function to preprocess mongo docs into
                                    an array of binary vectors
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Create a label binarizer to convert a single symbol string into a vector of binaries
        preprocess_coordcount, lb = self._coordcount(docs, return_lb=True)

        def preprocess_neighbors_coordcounts(docs):
            '''
            We do the same thing that we do in the `_coordcount` method, but for
            `neighborcoord` instead.
            '''
            # We'll be doing this action a lot. Let's wrap it.
            def _calc_coordcount(coord):
                ''' Turn a `coord` string into a list of integers, i.e., coordcount '''
                return np.sum(lb.transform(coord.split('-')), axis=0)
            # n_symbols is the total number of symbols we'll be considering. Used for initialization
            n_symbols = len(_calc_coordcount(''))

            # Package the calculation into a function so that we can parallelize it
            def _preprocess_neighbors_coordcounts(doc):
                ncoord = doc['neighborcoord']
                # Initialize `coordcount_array`, which will be an array of integers
                # whose rows will correspond to the cumulative coordcounts of each
                # respective element
                coordcount_array = np.zeros((n_symbols, n_symbols), dtype=np.int64)
                # Add the coordcount vector for each neighbor
                for coord_string in ncoord:
                    # Unpack the information and then calculate the coordcount for this neighbor
                    neighbor, coord = coord_string.split(':')
                    coordcount = _calc_coordcount(coord)
                    # Find the row within the array that this neighbor should map to
                    neighbors_index, = np.where(_calc_coordcount(neighbor) == 1)
                    # Add the coordcount to the array
                    coordcount_array[neighbors_index, :] += np.array(coordcount)
                # Flatten the array into a vector so that our regressors can use it,
                # and then add it to the output
                coordcount_vector = np.concatenate(coordcount_array, axis=0)
                return coordcount_vector
            # Use GASpy multiprocessing to do the calculation
            neighbors_coordcounts = utils.multimap(_preprocess_neighbors_coordcounts, docs)

            return np.array(neighbors_coordcounts)

        return preprocess_neighbors_coordcounts


    # TODO:  Create this method
    def _gcn(self, docs=None):
        raise Exception('This method has not been created yet')


    # TODO:  Create this method
    def _dband_center(self, docs=None):
        raise Exception('This method has not been created yet')


    def _hash(self, docs=None, excluded_fingerprints='default'):
        '''
        Create a preprocessing function that hashes all of the inputs and then binarizes them.
        This is meant to create a unique identifier for each site, which is useful for testing.

        Input:
            docs                    Mongo json dictionaries. Default value of `None`
                                    yields `self.docs`
            excluded_fingerprints   A list of strings for all of the fingerprints inside of
                                    `docs` that the user does not want to hash. If `None`,
                                    then we set it as an empty list. If 'default', then set
                                    it to ['energy', 'adsorbates'].
        Output:
            preprocess_hash     Function to turn mongo docs
        '''
        # Manage default arguments
        if not docs:
            docs = copy.deepcopy(self.docs)
        if not excluded_fingerprints:
            excluded_fingerprints = []
        elif excluded_fingerprints == 'default':
            excluded_fingerprints = ['energy', 'adsorbates']

        # Turn the list of excluded fingerprints into an empty dictionary so that we
        # can use the `in` statement later on
        excluded_fingerprints = dict.fromkeys(excluded_fingerprints)

        # Create a function to hash a doc, then hash all of them
        def __hash(doc):
            ''' Hash everything inside of a document, excluding the `excluded_fingerprints` '''
            _str = ''
            for fingerprint, value in doc.iteritems():
                if fingerprint not in excluded_fingerprints:
                    _str += fingerprint + '=' + str(value) + '; '
            return hash(_str)
        hashes = [__hash(doc) for doc in docs]
        # Now use the hashes to fit a binarizer
        lb = preprocessing.LabelBinarizer()
        lb.fit(hashes)

        # Create another function that hashes documents and then transforms them
        # with the binarizer we just made
        def preprocess_hash(docs):
            hashes = utils.multimap(__hash, docs)
            bin_hashes = utils.multimap(lb.transform, hashes)
            return bin_hashes

        return preprocess_hash


    def transform(self, docs):
        '''
        Turn mongo documents into preprocessed, stacked, and scaled inputs.

        Input:
            docs    Mongo json documents, i.e., a dictionary of lists of data
        Output:
            preprocessed_features   A numpy array. The first axis extends for len(docs),
                                    and the second axis contains all of the preprocessed,
                                    scaled features.
        '''
        # Preprocess, stack, and scale the features
        features = [preprocess(docs) for preprocess in self.preprocessors.values()]
        stacked_features = np.concatenate(tuple(features), axis=1)
        # TODO:  Figure out how to keep scaling without GP losing scaling terribly
        # scaled_features = self.scaler.transform(stacked_features)
        # return np.array(scaled_features)
        return stacked_features


    # TODO:  Create this method
    def revert(self, processed_features):
        '''
        This method will turn a vector of processed features back into a mongo doc (dictionary)
        of information.

        Input:
            preprocessed_features   A numpy array. The first axis extends for len(docs),
                                    and the second axis contains all of the preprocessed,
                                    scaled features.
        Output:
            docs    A mongo-style list of dictionaries
        '''
        pass
