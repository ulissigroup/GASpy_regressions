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
import tqdm
from sklearn import preprocessing, decomposition
from sklearn.linear_model import LinearRegression
import mendeleev
from pymatgen.matproj.rest import MPRester
from gaspy.utils import read_rc


class GASpyPreprocessor(object):
    '''
    We transform fingerprints into features by creating preprocessing functions
    for each of the features, and then concatenating the outputs of each of these functions
    into a single numpy array. The preprocessing functions are created in the hidden
    methods (i.e., ones that start with `_`).
    '''
    def __init__(self, docs, features, chem_fp_ads=None, dim_red=False, **kwargs):
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
            chem_fp_ads Some of our chemical fingerprinting features using the simulated
                        adsorption energies. So we need to know which adsorbate to
                        consider when doing this. This argument, which should be a string,
                        indicates which adsorbate you want to use when doing the
                        chemical fingerprint. Obviously, this argument only does anything
                        when one of your features involves chemical fingerprinting.
            dim_red     A string indicating which dimensionality reduction technique you want
                        to use. Currently accepts `False` for no dimensionality reduction
                        and 'truncated_svd'.
            kwargs      Whatever arguments you want to pass to the dimensionality reducer.
        Resulting attributes:
            docs            The same as the `docs` input, but simply saved as an attribute
            preprocessors   An ordered dictionary whose keys are features and whose
                            values are the sklearn.preprocessing classes for each feature
            scaler          An instance of the sklearn.preprocessing.StandardScaler class
                            that has been fit to the preprocessed, stacked `docs`
        '''
        # We save the training documents because they're used during... training. If we
        # starting running into [more] serious memory issues, we should consider deleting
        # these after training is done.
        self.docs = docs

        # If we're doing chemical fingerprinting, then pull out the mendeleev objects
        # for each element
        if any(['chemfp' in feature_name for feature_name in features]):
            # Find all of the unique substrate elements in our documents
            symbols = [doc['symbols'] for doc in docs]
            self.non_substrate_elements = set([chem_fp_ads, 'U', ''])
            all_elements = np.unique([element for sublist in symbols for element in sublist
                                      if element not in self.non_substrate_elements])
            # Pull out the appropriate mendeleev information for each element and save the
            # information into a class attribute.
            self.mendeleev_data = dict.fromkeys(all_elements)
            for el in all_elements:
                self.mendeleev_data[el] = getattr(mendeleev, el)
            # Assign the `chem_fp_ads` as an attribute for later use. This could be a
            # method argument, but we pass it as an attribute to keep the method-calling
            # portion of __init__ consistent across methods.
            self.chem_fp_ads = chem_fp_ads

        # Create the preprocessors for each feature
        self.preprocessors = OrderedDict.fromkeys(features)
        for feature in features:
            self.preprocessors[feature] = getattr(self, '_' + feature)()

        # Partially preprocess the fingerprints, then stack them into a numpy array
        numerical_features = [preprocess(docs) for preprocess in self.preprocessors.values()]
        stacked_features = np.concatenate(tuple(numerical_features), axis=1)

        # Create and fit the scaler
        self.scaler = preprocessing.StandardScaler()
        scaled_features = self.scaler.fit_transform(stacked_features.astype(float))

        # Create and fit the dimensionality reducer. We used both an if-statement and EAFP
        # because we want to throw an error when the user tries something wrong, but we
        # don't want to throw an error if they didn't ask for dimensionality reduction.
        if dim_red:
            try:
                self.dim_reduce = getattr(self, '_' + dim_red)(scaled_features, **kwargs)
            except AttributeError as error:
                error.message += '; the %s dim_red argument is not recognized' % dim_red
                raise
        else:
            self.dim_reduce = getattr(self, '_do_nothing')()

        # Delete the documents now that we're done with them. Because they're probably huge.
        del self.docs


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
            docs                Mongo json dictionaries over which to perform the fitting.
                                Default value of `None` yields `self.docs`
        Outputs:
            lb          The label binarizer used to turn the string into an array of binary counts
            coordcount  The actual vector of integers that are the preprocessed form of `docs`
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Pull out all of the symbols that we can find in `docs`
        symbols = [doc['symbols'] for doc in docs]

        # Find all of the unique symbols. Ten save
        unq_symbols = np.unique([item for sublist in symbols for item in sublist])
        # Then fit a label binarizer on them that will be able to convert a single string into a binary vector.
        lb = preprocessing.LabelBinarizer()
        lb.fit(unq_symbols)

        return lb


    def _do_nothing(self):
        '''
        Create a function that does nothing; meant to be used when you don't want
        to do dimensionality reduction but the pipelines expects one.
        '''
        def __do_nothing(inputs):
            return inputs
        return __do_nothing


    def _pca(self, training_data, **kwargs):
        '''
        Create a dimensionality reducing function that uses SKLearn's `PCA` class.

        Inputs:
            training_data   The data set that you want to train the dimensionality reducer on.
                            It should probably be a numpy vector.
            kwargs          Whatever arguments you want to pass to the dimensionality reducer.
        Output:
        '''
        # Use SKLearn's PCA
        self.dim_reducer = decomposition.PCA(**kwargs)
        self.dim_reducer.fit(training_data)

        # Create the function that the `transform` method will be using
        def pca(inputs):
            outputs = self.dim_reducer.transform(inputs)
            return outputs

        return pca


    def _truncated_svd(self, training_data, **kwargs):
        '''
        Create a dimensionality reducing function that uses SKLearn's `TruncatedSVD` class.

        Inputs:
            training_data   The data set that you want to train the dimensionality reducer on.
                            It should probably be a numpy vector.
            kwargs          Whatever arguments you want to pass to the dimensionality reducer.
        Output:
        '''
        # Use SKLearn's TruncatedSVD
        self.dim_reducer = decomposition.TruncatedSVD(**kwargs)
        self.dim_reducer.fit(training_data)

        # Create the function that the `transform` method will be using
        def truncated_svd(inputs):
            outputs = self.dim_reducer.transform(inputs)
            return outputs

        return truncated_svd


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
            # Package the calculation into a function so that we can possibly parallelize it
            def _preprocess_coordcount(doc):  # noqa: E306
                coord = doc['coordination']
                coordcount = np.sum(lb.transform(coord.split('-')), axis=0)
                return coordcount
            coordcounts = [_preprocess_coordcount(doc) for doc in docs]
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
            # Package the calculation into a function so that we can possibly parallelize it
            def _preprocess_rnnc_count(doc):  # noqa: E306
                ''' Yes, this is hacky. Please don't judge me. '''
                coord = doc['coordination']
                nnc = doc['nextnearestcoordination']
                nnc_count = np.sum(lb.transform(nnc.split('-')), axis=0)
                coordcount = preprocess_coordcount([{'coordination': coord}])
                rnnc_count = nnc_count - coordcount
                return rnnc_count.flatten()
            rnnc_counts = [_preprocess_rnnc_count(doc) for doc in docs]
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

        # Find all of the unique adsorbate types. Then fit a label binarizer on them
        # that will be able to convert a single string into a binary vector.
        ads = np.unique([doc['adsorbate'] for doc in docs])
        lb = preprocessing.LabelBinarizer()
        lb.fit(ads)

        # Create the preprocessing function
        def preprocess_ads(docs):
            # Package the calculation into a function so that we can possibly parallelize it
            def _preprocess_ads(doc):  # noqa: E306
                ads = doc['adsorbate']
                ads_vector = lb.transform(ads)[0]
                return ads_vector
            ads_vectors = [_preprocess_ads(doc) for doc in docs]
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

            # Package the calculation into a function so that we can possibly parallelize it
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
            neighbors_coordcounts = [_preprocess_neighbors_coordcounts(doc) for doc in docs]

            return np.array(neighbors_coordcounts)

        return preprocess_neighbors_coordcounts


    def __calculate_per_atom_energies(self, docs, doc_to_counts, lb):
        '''
        This method uses pure-metal adsorption data to calculate the average adsorption
        energy per binding atom. For example:  If the average CO binding energy on Cu
        is -0.24 eV for on-top sites, -0.51 eV for bridge sites, and -0.74 eV for
        3-fold sites, and -1.01 eV for 4-fold sites, then the average adsorption
        energy per binding atom for Cu would be -0.25 eV.

        Inputs:
            docs            Mongo json dictionaries.
            doc_to_counts   A function that can convert an adsorption site document
                            (the ones in `docs`) into a binary vector of counts for
                            each element. You should get this from the `_coordcount`
                            or `neighbors_coordcounts` methods.
            lb              The label binarizer used to help create the `doc_to_counts`
                            function
        Output:
            per_atom_energies   A nested dictionary whose keys are the adsorbates that
                                you want the energies for. The values are dictionaries
                                whose keys are the substate elements found in `docs`
                                and whose values are the average per-atom
                                adsorption energy for that element.
        '''
        docs = copy.deepcopy(docs)

        # Filter the documents to include only pure metals. We do this by first
        # finding compositions of all of the mpid's we're looking at and then
        # including only documents that contain one element in the composition
        mpids = set(doc['mpid'] for doc in docs)
        self.compositions_by_mpid = {}
        print('Beginning to pull data from the Materials Project...')
        with MPRester(read_rc()['matproj_api_key']) as m:
            for mpid in tqdm.tqdm(mpids):
                entry = m.get_entry_by_material_id(mpid)
                self.compositions_by_mpid[mpid] = entry.as_dict()['composition'].keys()
        docs = [doc for doc in docs if len(self.compositions_by_mpid[doc['mpid']]) == 1]

        # Now find all of the unique adsorbates that we've done pure metal calculations for.
        adsorbates = set(doc['adsorbate'] for doc in docs)

        # Calculate the per-atom-energies for each adsorbate type
        per_atom_energies = {}
        for ads in adsorbates:
            docs_subset = [doc for doc in docs if doc['adsorbate'] == ads]
            # Fitting a linear regressor that uses elemental counts to predict energies.
            counts = doc_to_counts(docs_subset)
            energies = np.array([doc['energy'] for doc in docs_subset])
            regressor = LinearRegression(fit_intercept=False)
            try:
                regressor.fit(counts, energies)
            except ValueError as error:
                error.args = (error.args[0] + '; you might not have supplied a `chem_fp_ads` argument',)
                raise

            # Pull out the coefficients of the regressor. These coefficients are the
            # "per atom adsorption energy". There is one energy per element, so we store
            # these data in a dictionary.
            per_atom_energies[ads] = {}
            n_elements = counts[0].shape[0]
            for i in range(n_elements):
                # We identify the element corresponding to this elemental index
                count = np.zeros((1, n_elements))
                count[0, i] = 1
                element = lb.inverse_transform(count)[0]
                # Then assign the coefficient to the dictionary
                per_atom_energies[ads][element] = regressor.coef_[i]

        return per_atom_energies


    def _pooled_coordatoms_chemfp0(self, docs=None):
        '''
        Create a preprocessing function to calculate the pooled chemical fingerprints of
        the substrate atoms that are coordinated with the adsorbate. The chemical fingerprint
        number 0 (chemfp0) of an atom consists of its per-atom-adsorption-energy, its atomic
        number, and its Pauling electronegativity. The "pooling" part comes in when we combine
        the chemfp0 of identical elements (e.g., Cu and Cu) into one vector and append a count.
        Note that since we have a variable number of features (i.e., coordination atoms), some
        parts of this vector will not have "valid" values to assign. In these cases, we
        will use the VNF method published by Davie et al (Kriging atomic properties with
        a variable number of inputs, J Chem Phys 2016).

        WARNING:  Since per-atom-adsorption-energy depends on the adsorbate, this feature
        should be exclusive with the `ads` feature. If you want to work with multiple
        adsorbates, then you should block by adsorbate.

        Example:  Pending

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_cfp0     Function to preprocess mongo docs into a vector of floats
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # This coordcount calculator will help us calculate per-atom-energies
        calc_coordcount, lb = self._coordcount(docs, return_lb=True)
        # Calculate the per-atom-adsorption-energies for each adsorbate-element pairing.
        per_atom_energies = self.__calculate_per_atom_energies(docs, calc_coordcount, lb)

        # For each adsorbate type, establish what a "blank" fingerprint would look like
        # as per the VNF rules. The item that we choose to set as "outside of valid range"
        # is the count.
        dummy_fps = {}
        max_comp = {}
        for adsorbate, energies_by_element in per_atom_energies.iteritems():
            avg_energy = np.average([energy for energy in energies_by_element.values()])
            avg_atomic_num = np.average([self.mendeleev_data[element].atomic_number
                                         for element in energies_by_element.keys()])
            avg_electroneg = np.average([self.mendeleev_data[element].electronegativity(scale='pauling')
                                         for element in energies_by_element.keys()])
            dummy_count = 0
            dummy_fps[adsorbate] = (avg_energy, avg_atomic_num, avg_electroneg, dummy_count)
            # Not all of our coordination sites will have the same number of substrate element
            # types. This means we'll have a variable length of fingerprints. We will address
            # this by filling any empty spaces, but before that we need to know how many total
            # spaces there should be. We figure that out here.
            mpids_subset = [doc['mpid'] for doc in docs if doc['adsorbate'] == adsorbate]
            compositions = [len(self.compositions_by_mpid[mpid]) for mpid in mpids_subset]
            max_comp[adsorbate] = max(compositions)

        def preprocess_pooled_coordatoms_chemfp0(docs):
            # This feature needs to know the adsorbate. Let's make sure that the adsorbate
            # key is there.
            if 'adsorbate' not in docs[0]:
                for doc in docs:
                    doc['adsorbate'] = doc['adsorbates'][0]

            # Package the calculation into a function so that we can possibly parallelize it
            def calc_pooled_coordatoms_chemfps(doc):  # noqa: E306
                # We first find both the adsorbate and the binding atoms, which we need
                # to do the chemical fingerprinting
                adsorbate = doc['adsorbate']
                binding_atoms = doc['coordination'].split('-')
                # Fingerprint each type of element that's present in the set of coordinated atoms
                elemental_fps = []
                for element in set(binding_atoms):
                    element_data = self.mendeleev_data[element]
                    energy = per_atom_energies[adsorbate][element]
                    atomic_number = element_data.atomic_number
                    electronegativity = element_data.electronegativity(scale='pauling')
                    count = binding_atoms.count(element)
                    elemental_fps.append((energy, atomic_number, electronegativity, count))
                # Now we sort the fingerprints. The per-atom-adsorption-energy is first in
                # the tuple of fingerprints, so that's what Python will sort on (from
                # lowest to highest)
                return sorted(elemental_fps)
            # Put the function through a comprehension. We don't necessarily need to do it this
            # way, but it'll be easy to multithread it later if we decide to do so.
            sparse_elemental_fps = [calc_pooled_coordatoms_chemfps(doc) for doc in docs]

            # Fill in the dummy features here
            chem_fps = []
            for doc, sparse_elemental_fp in zip(docs, sparse_elemental_fps):
                ads = doc['adsorbate']
                chem_fp = []
                for i in range(max_comp[ads]):
                    # EAFP to fill in the real fingerprints first and then the dummy ones
                    try:
                        chem_fp.append(sparse_elemental_fp[i])
                    except IndexError:
                        chem_fp.append(dummy_fps[ads])
                site_fp = np.array(chem_fp).flatten()
                chem_fps.append(site_fp)
            return np.array(chem_fps)

        return preprocess_pooled_coordatoms_chemfp0


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
                                    it to ['energy', 'adsorbates', 'adsorbate'].
        Output:
            preprocess_hash     Function to turn mongo docs
        '''
        # Manage default arguments
        if not docs:
            docs = copy.deepcopy(self.docs)
        if not excluded_fingerprints:
            excluded_fingerprints = []
        elif excluded_fingerprints == 'default':
            excluded_fingerprints = ['energy', 'adsorbates', 'adsorbate']

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
            hashes = [__hash(doc) for doc in docs]
            bin_hashes = lb.transform(hashes)
            return bin_hashes

        return preprocess_hash


    def transform(self, docs):
        '''
        Turn mongo documents into preprocessed, stacked, scaled, and reduced inputs.

        Input:
            docs    Mongo json documents, i.e., a dictionary of lists of data
        Output:
            preprocessed_features   A numpy array. The first axis extends for len(docs),
                                    and the second axis contains all of the preprocessed,
                                    scaled features.
        '''
        # Preprocess, stack, scale, and reduce the features
        features = tuple(preprocess(docs) for preprocess in self.preprocessors.values())
        stacked_features = np.concatenate(tuple(features), axis=1)
        scaled_features = self.scaler.transform(stacked_features.astype(float))
        preprocessed_features = self.dim_reduce(scaled_features)
        return preprocessed_features
