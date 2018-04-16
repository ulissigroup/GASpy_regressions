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
    def __init__(self, docs, features, dim_red=False, **kwargs):
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


    def __pull_mendeleev_data(self, docs=None):
        '''
        This method fetches Mendeleev data for all elements that we might want to look at.
        We do this here so that we only have to pull Mendeleev data once, thereby speeding
        things up.

        Input:
            docs    Mongo json dictionaries. Defaults to `self.docs`, which should be
                    the document supplied during initialization.
        Resulting attribute:
            mendeleev_data  A dictionary whose keys are the element names and whose
                            attributes are the mendeleev objects.
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # Find all of the unique substrate elements in our documents
        symbols = [doc['symbols'] for doc in docs]
        all_elements = set(element for sublist in symbols for element in sublist)
        # Pull out the appropriate mendeleev information for each element and save the
        # information into a class attribute.
        self.mendeleev_data = dict.fromkeys(all_elements)
        for el in all_elements:
            self.mendeleev_data[el] = getattr(mendeleev, el)


    def __calculate_median_adsorption_energies(self, docs):
        '''
        This method uses pure-metal adsorption data to calculate the median adsorption

        Inputs:
            docs    Mongo json dictionaries.
        Resulting attributes:
            compositions_by_mpid        A dictionary whose keys are mpid and whose values
                                        are lists of strings for each element in that MPID
            median_adsorption_energies  A nested dictionary whose keys are the adsorbates that
                                        you want the energies for. The values are dictionaries
                                        whose keys are the substate elements found in `docs`
                                        and whose values are the median adsorption energy
                                        for that element.
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

        # For each adsorbate, find the elements we've done calculations for
        self.median_adsorption_energies = {}
        for ads in adsorbates:
            docs_subset = [doc for doc in docs if doc['adsorbate'] == ads]
            elements = set([self.compositions_by_mpid[doc['mpid']][0] for doc in docs])

            # Calculate and store the median adsorption energy
            # for each adsorbate-element pairing
            self.median_adsorption_energies[ads] = {}
            for el in elements:
                energies = [doc['energy'] for doc in docs_subset
                            if self.compositions_by_mpid[doc['mpid']][0] == el]
                median = np.median(energies)
                # Sometimes our data is sparse and yields no energies to take medians on.
                # When this happens, just take the median of all elements.
                if np.isnan(median):
                    energies = [doc['energy'] for doc in docs_subset]
                    median = np.median(energies)
                self.median_adsorption_energies[ads][el] = median


    def __chemfp0_site(self, coord, ads, normalize=False, coord_num=1):
        '''
        This method will calculate what we call the "chemical fingerprint number 0"
        of a site. This fingerprint will be a Nx4 array where N is the number of unique
        types of elements that are coordinated with the site. The elements in each 1x4
        vector are the element's median adsorption energy, its atomic number, its
        Pauling electronegativity, and the number of those elements that show up in
        the coordination. We also sort the 1x4 vectors such that the first 1x4 vector
        that shows up is the one with the lowest median adsorption energy.

        Inputs:
            coord       A list of strings that represent the coordination of the site
                        you want to fingerprint. For example, a 3-fold copper site would be
                        ['Cu', 'Cu', 'Cu'].
            ads         This fingerprit uses a "median adsorption energy" (check out the
                        `__calculate_median_adsorption_energies` method for more details). Since
                        these values are adsorbate-specific, you need to tell us what
                        adsorbate you want to do the fingerprinting with.
            normalize   A boolean indicating whether you want to divide the elemental counts
                        by the coordination number.
            coord_num   An integer representing the coordination number. This is used
                        only when `normalize == True`.
        Output:
            chem_fp     A list of tuples. The length of the list is equal to the number
                        of elements present in the coordination, and the length of the
                        tuples is 4. The first value in each tuple is the median
                        adsorption energy; the second value is the atomic number;
                        the third value is the Pauling electronegativity; and the last
                        number is the count of that element in the coordination site.
        '''
        chem_fp = []

        # Just... do the calculation
        for element in set(coord):
            try:
                element_data = self.mendeleev_data[element]
                energy = self.median_adsorption_energies[ads][element]
                atomic_number = element_data.atomic_number
                electronegativity = element_data.electronegativity(scale='pauling')
                count = coord.count(element)
                if normalize:
                    count /= float(coord_num)
                chem_fp.append((energy, atomic_number, electronegativity, count))
            # For some silly cases, there are no coordinated atoms. When that
            # happens, just pass out the dummy fingerprint. Use EAFP just in
            # case we've never actually calculated the dummy fingerprint yet.
            except KeyError:
                try:
                    chem_fp.append(self.dummy_fps[ads])
                except AttributeError:
                    self.__define_dummy_chemfp0()
                    chem_fp.append(self.dummy_fps[ads])

        return sorted(chem_fp)


    def __define_dummy_chemfp0(self, docs=None):
        '''
        This method establishes a "dummy" value for the `chemfp0` type of feature.
        This dummy value is useful when using variable number of features;
        reference Davie et al (Kriging atomic properties with a variable number of inputs,
        J Chem Phys 2016). The out-of-bounds feature we choose is the atomic count.

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Resulting attributes:
            dummy_fps   A dictionary whose keys are the adsorbate you want a dummy
                        feature for and whose keys are a tuple (i.e., the chemfp0)
            max_comp    A dictionary whose keys are the adsorbate you are considering
                        and whose values are the maximum number of elements present
                        in any single mpid we are looking at. This is useful for
                        figuring out how many dummy features you need to add.
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # This method requires the median adsorption energy we call the pertinent method
        # that will calculate these values and assign them as attributes. We only do this
        # if it looks like this method has not been called already. Then do the same
        # for Mendeleev data.
        if not hasattr(self, 'median_adsorption_energies'):
            self.__calculate_median_adsorption_energies(docs)
        if not hasattr(self, 'mendeleev_data'):
            self.__pull_mendeleev_data(docs)

        # Calculate `dummy_fps`
        self.dummy_fps = {}
        self.max_comp = {}
        for adsorbate, energies_by_element in self.median_adsorption_energies.iteritems():
            avg_energy = np.average([energy for energy in energies_by_element.values()])
            avg_atomic_num = np.average([self.mendeleev_data[element].atomic_number
                                         for element in energies_by_element.keys()])
            avg_electroneg = np.average([self.mendeleev_data[element].electronegativity(scale='pauling')
                                         for element in energies_by_element.keys()])
            dummy_count = 0
            self.dummy_fps[adsorbate] = (avg_energy, avg_atomic_num, avg_electroneg, dummy_count)

            # Calculate `max_comp`
            mpids_subset = [doc['mpid'] for doc in docs if doc['adsorbate'] == adsorbate]
            compositions = [len(self.compositions_by_mpid[mpid]) for mpid in mpids_subset]
            self.max_comp[adsorbate] = max(compositions)


    def _coordatoms_chemfp0(self, docs=None):
        '''
        Create a preprocessing function to calculate the chemical fingerprints of
        the substrate atoms that are coordinated with the adsorbate (reference the
        `__chemfp0_site` method for more details). This feature also uses the VNF
        methodology (reference the `__define_dummy_chemfp0` method).

        WARNING:  Since median adsorption energy depends on the adsorbate, this feature
        should be exclusive with the `ads` feature. If you want to work with multiple
        adsorbates, then you should block by adsorbate. Why is it exclusive, you ask?
        Because I'm too lazy to code it.

        Example:  Pending

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_coord_chemfp0    Function to preprocess mongo docs into a vector of floats
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # We need to calculate the median adsorption energies and also establish some dummy features.
        # These two methods do this and assign them as attributes. Note the `if-then` that
        # makes sure we only call these methods if they have not yet been executed.
        if not hasattr(self, 'dummy_pfs'):
            self.__define_dummy_chemfp0(docs)
        if not hasattr(self, 'median_adsorption_energies'):
            self.__calculate_median_adsorption_energies(docs)

        def preprocess_coord_chemfp0(docs):
            # This feature needs to know the adsorbate. Let's make sure that the adsorbate
            # key is there.
            if 'adsorbate' not in docs[0]:
                for doc in docs:
                    doc['adsorbate'] = doc['adsorbates'][0]

            # Package the calculation into a function so that we can possibly parallelize it
            def calc_coordatoms_chemfps(doc):  # noqa: E306
                # We first find both the adsorbate and the binding atoms, which we then
                # use to do the chemical fingerprinting
                adsorbate = doc['adsorbate']
                binding_atoms = doc['coordination'].split('-')
                elemental_fps = self.__chemfp0_site(binding_atoms, adsorbate)
                return elemental_fps
            # Put the function through a comprehension. We don't necessarily need to do it this
            # way, but it'll be easy to multithread it later if we decide to do so.
            sparse_elemental_fps = [calc_coordatoms_chemfps(doc) for doc in docs]

            # Fill in the dummy features here
            chem_fps = []
            for doc, sparse_elemental_fp in zip(docs, sparse_elemental_fps):
                ads = doc['adsorbate']
                chem_fp = []
                for i in range(max(self.max_comp.values())):
                    # EAFP to fill in the real fingerprints first and then the dummy ones
                    try:
                        chem_fp.append(sparse_elemental_fp[i])
                    except IndexError:
                        chem_fp.append(self.dummy_fps[ads])
                site_fp = np.array(chem_fp).flatten()
                chem_fps.append(site_fp)
            return np.array(chem_fps)

        return preprocess_coord_chemfp0


    def _neighbors_chemfp0(self, docs=None):
        '''
        Create a preprocessing function to calculate the chemical fingerprints of the
        second-shell atoms, i.e., the ones bonded to the coordination atoms (reference
        the `__chemfp0_site` method for more details). This feature also uses the VNF
        methodology (reference the `__define_dummy_chemfp0` method). Note that this specific
        incarnation of chemfp0 has two subtle nuances:  1) for ease-of-programming reasons,
        we end up multi-counting neighbors that are bonded to more than one coordinated atom,
        and 2) we divide each elemental count by the coordination number of the adsorbate.

        WARNING:  Since median adsorption energy depends on the adsorbate, this feature
        should be exclusive with the `ads` feature. If you want to work with multiple
        adsorbates, then you should block by adsorbate. Why is it exclusive, you ask?
        Because I'm too lazy to code it.

        Example:  Pending

        Input:
            docs    Mongo json dictionaries. Default value of `None` yields `self.docs`
        Output:
            preprocess_neighbor_chemfp0  Function to preprocess mongo docs into a vector of floats
        '''
        if not docs:
            docs = copy.deepcopy(self.docs)

        # We need to calculate the median adsorption energy establish some dummy features.
        # These two methods do this and assign them as attributes. Note the `if-then` that
        # makes sure we only call these methods if they have not yet been executed.
        if not hasattr(self, 'median_adsorption_energies'):
            self.__calculate_median_adsorption_energies(docs)
        if not hasattr(self, 'dummy_pfs'):
            self.__define_dummy_chemfp0(docs)

        def preprocess_neighbor_chemfp0(docs):
            # This feature needs to know the adsorbate. Let's make sure that the adsorbate
            # key is there.
            if 'adsorbate' not in docs[0]:
                for doc in docs:
                    doc['adsorbate'] = doc['adsorbates'][0]

            # Package the calculation into a function so that we can possibly parallelize it
            def calc_coordatoms_chemfps(doc):  # noqa: E306
                # We first find the adsorbate, the coordination number, and
                # all of the neighbors.
                adsorbate = doc['adsorbate']
                # coord_num = len(doc['coordination'].split('-'))
                all_neighbors = []
                for coord_string in doc['neighborcoord']:
                    binding_atom, neighbors = coord_string.split(':')
                    all_neighbors.extend(neighbors.split('-'))
                # Calculate the chemical fingerprint
                elemental_fps = self.__chemfp0_site(all_neighbors, adsorbate)
                return elemental_fps
            # Put the function through a comprehension. We don't necessarily need to do it this
            # way, but it'll be easy to multithread it later if we decide to do so.
            sparse_elemental_fps = [calc_coordatoms_chemfps(doc) for doc in docs]

            # Fill in the dummy features here
            chem_fps = []
            for doc, sparse_elemental_fp in zip(docs, sparse_elemental_fps):
                ads = doc['adsorbate']
                chem_fp = []
                for i in range(max(self.max_comp.values())):
                    # EAFP to fill in the real fingerprints first and then the dummy ones
                    try:
                        chem_fp.append(sparse_elemental_fp[i])
                    except IndexError:
                        chem_fp.append(self.dummy_fps[ads])
                site_fp = np.array(chem_fp).flatten()
                chem_fps.append(site_fp)
            return np.array(chem_fps)

        return preprocess_neighbor_chemfp0


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
