'''
This class contains different methods to pull and parse data from *.db files so that
other scripts may perform data analyses on them.

Note that this script uses the term "factors". Some people may call these "features" or
"independent variables".

All of the non-hidden methods in this class return the same outputs:
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
    pp          A dictionary containing the preprocessors used when converting the
                fingerprints to features. The key is the name of the fingerprint.
    norms       An np.array (vector) that is the `norm` vector returned by
                sklearn.preprocessing.normalize. You need to divide inputs by `norms`
                for the prediction to work.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import pdb
import warnings
import sys
import pickle
import copy
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from vasp.mongo import mongo_doc_atoms
sys.path.insert(0, '../')
from gaspy import utils
from gaspy import defaults


class PullFeatures(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 vasp_settings=None,
                 energy_min=-4, energy_max=4, f_max=0.5,
                 ads_move_max=1.5, bare_slab_move_max=0.5, slab_move_max=1.5,
                 train_size=0.75, random_state=42):
        '''
        Connect to the database and pull out the pertinent docs/information in a form that
        may be regressed/manipulated.

        Inputs:
            vasp_settings       A string of vasp settings. Use the vasp_settings_to_str
                                function in GAspy
            energy_min          The minimum adsorption energy to pull from the Local DB (eV)
            energy_max          The maximum adsorption energy to pull from the Local DB (eV)
            ads_move_max        The maximum distance that an adsorbate atom may move (angstrom)
            bare_slab_move_max  The maxmimum distance that a slab atom may move when it is relaxed
                                without an adsorbate (angstrom)
            slab_move_max       The maximum distance that a slab atom may move (angstrom)
            f_max               The upper limit on the maximum force on an atom in the system
        '''
        # The default value for `vasp_settings` will be `None`, which means that we take all
        # calculations and do not filter via vasp settings. We define the default here
        # just in case Python doesn't play well with this non-straight-forward default.
        if not vasp_settings:
            vasp_settings = utils.vasp_settings_to_str({})

        # Create a function that returns a mongo cursor object given the arguments passed
        # to GASPull
        with utils.get_adsorption_db() as client:
            self.client = client

        # Pass along various parameters to use later
        self.train_size = train_size
        self.random_state = random_state
        self.vasp_settings = vasp_settings
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.f_max = f_max
        self.ads_move_max = ads_move_max
        self.bare_slab_move_max = bare_slab_move_max
        self.slab_move_max = slab_move_max

        # Do a quick `_pull` test to see if we even get anything. If not, then alert the user.
        docs = self._pull(defaults.fingerprints())
        if not len(docs[docs.keys()[0]]):
            raise Exception('GASPull failed to find any matches. Please check your query settings.')


    def _pull(self, fingerprints,
              adsorbates='default', vasp_settings='default',
              energy_min='default', energy_max='default', f_max='default',
              ads_move_max='default', bare_slab_move_max='default',
              slab_move_max='default'):
        '''
        This method really only calls `gaspy.utils.get_docs`, but it does so
        with some extra defaults to make it easier for the programmer to make new methods
        & not worry about a lot of the arguments that `get_docs` needs.

        Note that `fingerprints` should be mongo queries of parameters that you want
        pulled.

        Output:
            p_docs  "parsed docs"; a dictionary whose keys are the keys of `fingerprints` and
                    whose values are lists of the results returned by each query within
                    `fingerprints`.
        '''
        # It turns out that Python doesn't like `self.*` assigned as defaults.
        # Let's hack it. Note that we use the `default` flag instead of the
        # conventional `None` because the user might want to actually pass a `None`
        # to this method.
        if energy_min == 'default':
            energy_min = self.energy_min
        if energy_max == 'default':
            energy_max = self.energy_max
        if f_max == 'default':
            f_max = self.f_max
        if ads_move_max == 'default':
            ads_move_max = self.ads_move_max
        if bare_slab_move_max == 'default':
            bare_slab_move_max = self.bare_slab_move_max
        if slab_move_max == 'default':
            slab_move_max = self.slab_move_max
        if vasp_settings == 'default':
            vasp_settings = self.vasp_settings

        # Make sure that we are storing the mongo ID number in all of our `p_docs`
        fingerprints['mongo_id'] = '$_id'

        return utils.get_docs(self.client, 'adsorption', fingerprints,
                              adsorbates=None, calc_settings=None,
                              vasp_settings=vasp_settings,
                              energy_min=energy_min, energy_max=energy_max,
                              f_max=f_max, ads_move_max=ads_move_max,
                              bare_slab_move_max=bare_slab_move_max,
                              slab_move_max=slab_move_max)[1]


    def _coord2coordcount(self, coords):
        '''
        Turn a human-readable string coordination into a vector of
        coordination count.

        Inputs:
            coords      A list of strings indicating the coordination, e.g.,
                        ['Ag-Au-Au', 'Au']
        Outputs:
            coordcount  A numpy array of coordination counts, e.g.,
                        np.array([1 2 0 0 0], [0 1 0 0 0])
            lb          The label binarizer used to turn the strings into
                        lists of vectors
        '''
        # Get ALL of the symbols for the elements that we have in the entire DB
        symbols = self._pull({'symbols': '$atoms.chemical_symbols'},
                             vasp_settings=None, energy_min=None, energy_max=None,
                             f_max=None, ads_move_max=None, bare_slab_move_max=None,
                             slab_move_max=None)['symbols']
        # Parse the symbols to find the unique ones.
        unq_symbols = np.unique([item for sublist in symbols for item in sublist
                                 if (item != 'C' and item != 'O')])
        # TODO:  Get rid of C & O filter if we end up using carbides/oxides. We filter
        # because Alamo doesn't like unused features.

        # Make a binarizer out of all the symbols.
        # We filter out C & O because Alamo cries if we include
        # symbols that do not actually end up as part of the
        # coordination. We can put these back in if we start dealing
        # with carbides or oxides, respectively.
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.unique([item for sublist in symbols for item in sublist
                          if item != 'C' and item != 'O']))

        # Use the binarizer to turn a coordination into a list of sparse
        # binary vectors. For example: We could turn `Ag-Au-Au` into
        # [[1 0 0 0 0], [0 1 0 0 0], [0 1 0 0 0]]. We then use np.sum
        # to summate each list into a count list, [1 2 0 0 0]
        coordcount = np.array([np.sum(lb.transform(coord.split('-')), axis=0)
                               for coord in coords])
        return coordcount, lb


    def _stack(self, data_dict, keys):
        '''
        This method stacks various data into a numpy array. It should be used to stack factors
        and responses (separately).
        Inputs:
            data_dict   A dictionary of the data to stack. The keys in this dictionary should
                        be the names of the data, and the values should be lists of the data
            keys        `data_dict` may contain key:value pairs that we don't want to stack.
                        The user must provide a list of the keys of the data they want to stack.
        Output:  A numpy array where the data are stacked horizontally.
        '''
        # We will use the numpy.hstack function, which accepts the data in a tuple form. But first,
        # we initialize the tuple and then populate it with the data from data_dict.
        data_tup = tuple()
        for key in keys:
            data_tup += (data_dict[key],)
        return np.hstack(data_tup)


    def _post_process(self, features, factors, responses, p_docs):
        '''
        This method will stack all features into a single vector; stack all of the
        responses into a single vector, and then split the resulting data into training,
        testing, and train+test sets.

        Inputs:
            features    A pre-processed form of `p_docs`, which is a parsed set of mongo
                        documents whose keys are fingerprint names and values are the processed
                        values of the fingerprint.
            factors     A list of strings indicating which of the keys in `features`
                        should correspond to model factors
            responses   A list of strings indicating which of the keys in `features`
                        should correspond to model responses
            p_docs      The raw `p_docs` returned by the `self._pull` method
        Outputs:
            x               A dictionary whose keys are 'train', 'test', and 'train+test'.
                            The values are stacked numpy arrays of the model inputs.
            y               A dictionary whose keys are 'train', 'test', and 'train+test'.
                            The values are stacked numpy arrays of the model responses.
            split_p_docs    A re-structured version of `p_docs`. Now it is a dictionary
                            with the same keys as the other outputs. The values have the same
                            structure as the original `p_docs`.
            norms           An np.array (vector) that is the `norm` vector returned by
                            sklearn.preprocessing.normalize. You need to divide inputs by `norms`
                            for the prediction to work.
        '''
        # Initialize the outputs
        x = {'train': None, 'test': None, 'train+test': None}
        y = copy.deepcopy(x)
        split_p_docs = copy.deepcopy(x)

        # Stack p_docs to create the inputs, then normalize them
        x['train+test'], norm = preprocessing.normalize(self._stack(features, factors),
                                                        axis=0,
                                                        return_norm=True)
        # Stack p_docs to create the outputs
        y['train+test'] = self._stack(features, responses)
        # Split the inputs and outputs. We also pull out indices for splitting so that we
        # can split `p_docs` later
        x['train'], x['test'], y['train'], y['test'], indices_train, indices_test = \
                train_test_split(x['train+test'], y['train+test'], range(len(x['train+test'])),
                                 train_size=self.train_size, random_state=self.random_state)

        # Split/re-structure `p_docs`
        split_p_docs = {'train+test': p_docs,
                        'train': {fp: np.array(values)[indices_train] for fp, values in p_docs.iteritems()},
                        'test': {fp: np.array(values)[indices_test] for fp, values in p_docs.iteritems()},}

        return x, y, split_p_docs, norm


    def energy_fr_coordcount(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['coordination']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'coordination': '$processed_data.fp_final.coordination',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses)

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the coordination
        features['coordination'], pp['coordination'] = self._coord2coordcount(p_docs['coordination'])

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    def energy_fr_coordcount_ads(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            ads             A vector of binaries that indicate the type of adsorbate.
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['coordination', 'adsorbate']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'coordination': '$processed_data.fp_final.coordination',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses)

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(p_docs['adsorbate'])
        pp['adsorbate'] = preprocessing.LabelBinarizer()
        pp['adsorbate'].fit(ads)
        features['adsorbate'] = pp['adsorbate'].transform(p_docs['adsorbate'])
        # Pre-process the coordination
        features['coordination'], pp['coordination'] = self._coord2coordcount(p_docs['coordination'])

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    def energy_fr_structure_hash(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            ads             A vector of binaries that indicate the type of adsorbate.
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['hash']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'top': '$processed_data.calculation_info.top',
                        'coordination': '$processed_data.fp_final.coordination',
                        'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination',
                        'neighborcoord': '$processed_data.fp_init.neighborcoord',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses)

        pp = {}
        pp['hash'] =  preprocessing.LabelBinarizer()
        hashed_vals=map(lambda a,b,c,d,e,f: ''.join(map(str,[a,b,c,d,e,f])),p_docs['mpid'],p_docs['miller'],p_docs['coordination'],p_docs['nextnearestcoordination'],p_docs['top'],p_docs['neighborcoord'])
        pp['hash'] = pp['hash'].fit(hashed_vals)
        features['hash'] = pp['hash'].transform(hashed_vals)

        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    def energy_fr_coordcount_nncoord_ads(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            nncord          The same as coord_count, except we cound the atoms that are
                            coordinated with the binding atoms' nearest neighbor
            ads             A vector of binaries that indicate the type of adsorbate.
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['coordination', 'nextnearestcoordination', 'adsorbate']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'coordination': '$processed_data.fp_final.coordination',
                        'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses)

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(p_docs['adsorbate'])
        pp['adsorbate'] = preprocessing.LabelBinarizer()
        pp['adsorbate'].fit(ads)
        features['adsorbate'] = pp['adsorbate'].transform(p_docs['adsorbate'])
        # Pre-process the coordination counts
        features['coordination'], pp['coordination'] = self._coord2coordcount(p_docs['coordination'])
        # Pre-process the next nearest coordination counts
        # Then Remove the coordination, since it's redundant.
        features['nextnearestcoordination'], pp['nextnearestcoordination'] = \
                self._coord2coordcount(p_docs['nextnearestcoordination'])
        features['nextnearestcoordination'] = \
                features['nextnearestcoordination'] - features['coordination']

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    def energy_fr_coordcount_nncoord(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            nncord          The same as coord_count, except we cound the atoms that are
                            coordinated with the binding atoms' nearest neighbor
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['coordination', 'nextnearestcoordination']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'top': '$processed_data.calculation_info.top',
                        'coordination': '$processed_data.fp_final.coordination',
                        'nextnearestcoordination': '$processed_data.fp_final.nextnearestcoordination',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses)

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the coordination counts
        features['coordination'], pp['coordination'] = self._coord2coordcount(p_docs['coordination'])
        # Pre-process the next nearest coordination counts.
        # Then Remove the coordination, since it's redundant.
        features['nextnearestcoordination'], pp['nextnearestcoordination'] = \
                self._coord2coordcount(p_docs['nextnearestcoordination'])
        features['nextnearestcoordination'] = \
                features['nextnearestcoordination'] - features['coordination']

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    def energy_fr_nncoord(self):
        '''
        Pull data according to the following motifs:
            nncoord     The same as coord_count, except we count the atoms that are
                        coordinated with the binding atoms' nearest neighbor
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['nextnearestcoordination']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'coordination': '$processed_data.fp_final.coordination',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed such that they may be accepted
        # and readable by regressors
        features = dict.fromkeys(factors+responses+['coordination'])

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the coordination counts for the sole purpose of removing
        # them from the nncoord vector
        features['coordination'], pp['coordination'] = self._coord2coordcount(p_docs['coordination'])
        # Pre-process the next nearest coordination counts
        # Then Remove the coordination, since it's redundant.
        features['nextnearestcoordination'], pp['nextnearestcoordination'] = \
                self._coord2coordcount(p_docs['nextnearestcoordination'])
        features['nextnearestcoordination'] = \
                features['nextnearestcoordination'] - features['coordination']

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm


    # TODO:  Convert to aux db format... and pretty much re-do it
    def energy_fr_gcn_ads(self):
        # pylint: disable=too-many-statements, too-many-branches
        '''
        Pull data according to the following motifs:
            gcn     Generalized Coordination Number for an adsorbate on an alloy.
                    It is similar to the coordination count, but instead of simply
                    adding `1` for each neighbor, we add the fractional coordination
                    for that neighbor (i.e., if that neighbor normally has a CN of 12
                    and now it has a CN of 6, we assign its fractional coordination as
                    6/12=0.5). Note that this method has two shortfalls:
                        1) The fractional coordination of the neighbors is insensitive
                           to the identity of the missing next-neighbors,
                           e.g., if a bulk Ag slab atom is normally coordinated
                           as Ni-Ni-Ga-Ga, then a slab Ag atom that is
                           coordinated as Ni-Ni-Ga will look the same as
                           another slab Ag atom that is coordinated as Ni-Ga-Ga.
                        2) This algorithm does not differentiate the "normal
                           coordination" of one elemental atom in the bulk from
                           another. For example:  If an adsorbate coordination
                           is Ag-Ag and one of those Ag lost all of its neighbors
                           (i.e., had fractional coordination=0) while the other
                           Ag lost none (i.e., fractional coordination=1), then
                           this method would mark the adsorbate's gcn=0+1=1.
                           This would be the same gcn as a system whose two
                           Ag atoms had fractional coordinations of 0.5 (each).
            ads     A vector of binaries that indicate the type of adsorbate.
        Outputs:
            p_docs      "parsed mongo docs"; a dictionary containing the data we pulled from
                        the adsorption database. This object is taken raw from the _pull method.
            x           A stacked array containing all of the data for all of the factors
            y           A stacked array containing all of the data for all of the responses
            x_train     A subset of `x` intended to use as a training set
            y_train     A subset of `y` intended to use as a training set
            x_test      A subset of `x` intended to use as a validation set
            y_test      A subset of `y` intended to use as a validation set
            pp          A dictionary containing the preprocessors used when converting the
                        fingerprints to features. The key is the name of the fingerprint.
        '''
        # Identify the factors & responses. This will be used to build the outputs.
        factors = ['gcn', 'adsorbate']
        responses = ['energy']
        # Establish the variables to pull (i.e., `fingerprints`) and pull it from
        # the database
        fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                        'miller': '$processed_data.calculation_info.miller',
                        'coordination': '$processed_data.fp_final.coordination',
                        'neighborcoord': '$processed_data.fp_final.neighborcoord',
                        'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                        'bulkfwid': '$processed_data.FW_info.bulk',
                        'energy': '$results.energy'}
        p_docs = self._pull(fingerprints=fingerprints)
        # `adsorbates` returns a list of adsorbates. We're only looking at one right now,
        # so we pull it out into its own key.
        p_docs['adsorbate'] = [adsorbates[0] for adsorbates in p_docs['adsorbates']]

        # Initialize a second dictionary, `features`, that will be identical to the `p_docs`
        # dictionary, except the values will be pre-processed p_docs, not "raw" p_docs.
        features = dict.fromkeys(factors+responses, None)

        pp = {}
        # Pre-process the energy
        features['energy'] = np.array(p_docs['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(p_docs['adsorbate'])
        pp['adsorbate'] = preprocessing.LabelBinarizer()
        pp['adsorbate'].fit(ads)
        features['adsorbate'] = pp['adsorbate'].transform(p_docs['adsorbate'])

        # Create a binarizer for the elements we are looking at so that we can use
        # it to calculate the GCNs
        pp['coordination'] = self._coord2coordcount([''])[1]
        # Determine the maximum coordination of each of the elements in the bulk, c_max.
        # This will also be used to calculate the GCNs. Note that `cmax` will be a nested
        # dictionary. The highest level keys are the fwids of the bulks; the second level
        # keys (which are the first-level values) are the unique elements of that bulk;
        # and the second-level values are naive coordination numbers (i.e., they treat
        # all elements the same).
        with utils.get_aux_db() as aux_db:
            # `bulks` is a dictionary of all of the bulks we've relaxed, where
            # the key is the fwid and the value is the ase.Atoms object.
            bulks = {fwid: mongo_doc_atoms(aux_db.find({'fwid': fwid})[0])
                     for fwid in np.unique(p_docs['bulkfwid'])}
        cmax = dict.fromkeys(bulks)
        for fwid, bulk in bulks.iteritems():
            # PyMatGen prep-work before we calculate the coordination count
            # of each atom in the bulk. `neighbor_coords` a dictionary the coordination
            # numbers for each atom in the bulk, but sorted such that each key
            # is a different element and each value is a list of coordination numbers
            # (for each occurence of that element in the bulk).
            struct = AseAtomsAdaptor.get_structure(bulk)
            vcf = VoronoiCoordFinder(struct, allow_pathological=True)
            bulk_symbols = np.unique(bulk.get_chemical_symbols())
            neighbor_coords = dict.fromkeys(bulk_symbols, [])
            for i, atom in enumerate(bulk):
                # We use a try/except block to address QHull errors. Since we can't
                # figure out how to define a QHull error, we put down a blanket
                # exception and spit it out so the user will know what happened.
                try:
                    neighbor_sites = vcf.get_coordinated_sites(i, 0.8)
                    neighbor_atoms = [neighbor_site.species_string
                                      for neighbor_site in neighbor_sites]
                    neighbor_coords[atom.symbol].append(np.sum(pp['coordination'].transform(neighbor_atoms)))
                except Exception as ex:         # pylint: disable=broad-except
                    template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    print('We might have gotten a Qhull error at fwid=%s, atom #%s (%s)' \
                          % (fwid, i, atom.symbol))
                    # If we get an error, then just put a zero and move on because,
                    # well, we need to put something.
                    neighbor_coords[atom.symbol].append(0)
            # Use `neighbor_coords` to populate `cmax`. We also make it a float
            # so that we don't have to do it later (i.e., do it up front).
            cmax[fwid] = {symbol: float(max(neighbor_coord))
                          for symbol, neighbor_coord in neighbor_coords.iteritems()}

        # Pre-process the GCNs. But first we initialize and then turn the GCN
        # array into a float array, since we'll be adding fractional coordination
        # numbers
        features['gcn'], __lb = self._coord2coordcount(['']*len(p_docs['energy']))
        features['gcn'] = features['gcn'].astype(float)
        for i, coord in enumerate(p_docs['coordination']):
            # `p_docs['neighborcoord'][i]` contains the coordinations as a list of strings,
            # e.g., ['W:N-N', 'W:N-N-N']. Here, we split off the neighbor (i.e., the symbols
            # before ":") and then count the number of symbols within each coordination,
            # which we then assign as `neighbor_cn`, which is the conventional coordination
            # number of each of the adsorbates' neighbors.
            neighbor_cn = [len(neighbor_coord.split(':')[1].split('-'))
                           for neighbor_coord in p_docs['neighborcoord'][i]]
            # Calculate and assign the gcn contribution for each neighbor
            for j, neighbor in enumerate(coord.split('-')):
                try:
                    _cmax = cmax[p_docs['bulkfwid'][i]][neighbor]
                    # The index of `neighbor`'s element within the coordcount vector
                    index = self._coord2coordcount([neighbor])[0].tolist()[0].index(1)
                    try:
                        # Add the gcn contribution from this neighor.
                        features['gcn'][i][index] += neighbor_cn[j]/_cmax
                    # If _cmax somehow turns out to be zero, then add a gcn of
                    # 1 (if the neighbor is coordinated at all)
                    except ZeroDivisionError:
                        if neighbor_cn[j]:
                            features['gcn'][i][index] += 1
                # If the `coord` ends up containing an element that's not
                # in the bulk (which only happens on edge cases where
                # PyMatGen fails us), then simply set the neighbor_cn to 1
                # (if the neighbor is coordinated at all).
                except KeyError:
                    if neighbor_cn[j]:
                        features['gcn'][i][index] += 1
            try:
                if not i % 100:
                    print("Pulling out p_docs point #%s for GCN" % i)
            except ZeroDivisionError:
                pass

        # Stack, split, and structure the data
        x, y, p_docs, norm = self._post_process(features, factors, responses, p_docs)

        return x, y, p_docs, pp, norm
