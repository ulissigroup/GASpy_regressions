'''
This class contains different methods to pull and parse data from *.db files so that
other scripts may perform data analyses on them.

Note that this script uses the term "factors". Some people may call these "features" or
"independent variables".

All of the non-hidden methods in this class return the same outputs:
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
from preprocessor import GASpyPreprocessor


class FeaturePuller(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, vasp_settings=None,
                 energy_min=-4, energy_max=4, f_max=0.5,
                 ads_move_max=1.5, bare_slab_move_max=0.5, slab_move_max=1.5,
                 train_size=0.75, random_state=42):
        '''
        Establish various filters to use when pulling information out.

        Inputs/attributes:
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
        self.vasp_settings = vasp_settings
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.ads_move_max = ads_move_max
        self.bare_slab_move_max = bare_slab_move_max
        self.slab_move_max = slab_move_max
        self.f_max = f_max
        self.train_size = train_size
        self.random_state = random_state


    def preprocess_and_split(self, features, responses, fingerprints=None, collection='adsorption'):
        '''
        Pull data from the database, preprocess it as per `GASpyPreprocessor`,
        then split the data into training and test sets.

        Inputs:
            features        A list of strings for each of the features that you want to include.
                            These strings should correspond to the 1st-level hidden methods
                            in `GASpyPreprocessor`, but without the leading underscore.
                            For example:  features = ('coordcount', 'ads')
            features        A list of strings for each of the responses that you want to include.
                            Pretty much just like features.
            fingerprints    Mongo queries of parameters that you want pulled. Note that we
                            automatically set some of these queries based on the features
                            and responses you are trying to use. So you only need to define
                            mongo queries for any extra information you want.
            collection      A string for the mongo db collection you want to pull from.
        Outputs:
            x               A dictionary of stacked arrays containing the data for all of the
                            factors. The keys are 'train', 'test', and 'train+test', and they
                            correspond to the training set, test/validation set, and the
                            cumulation of the training/test sets.
            y               The same as `x`, but for the outputs, not the inputs
            split_p_docs    The same as `x`, but the values are instead parsed mongo docs
                            whose structures are similar to the ones returned by
                            `gaspy.utils.get_docs`. Except the values are now numpy arrays
                            instead of lists.
            pp              The instance of GASpyPreprocessor that was used to preprocess the
                            data set pulled by this method. This is used to preprocess other data
                            to make future predictions.
        '''
        # Python doesn't like dictionaries being used as default values, so we initialize here
        if not fingerprints:
            fingerprints = {}
        # Make sure that we are always pulling out/storing the mongo ID number
        fingerprints['mongo_id'] = '$_id'

        # Some features require specific fingerprints. Here, we make sure that those
        # fingerprints are included
        _features = dict.fromkeys(features)
        if 'coordcount' in _features:
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['symbols'] = '$atoms.chemical_symbols'
        if 'rnnc_count' in _features:
            fingerprints['coordination'] = '$processed_data.fp_final_coordination'
            fingerprints['symbols'] = '$atoms.chemical_symbols'
            fingerprints['nnc'] = '$processed_data.fp_init.nextnearestcoordination'
        if 'ads' in _features:
            fingerprints['adsorbates'] = '$processed_data.calculation_info.adsorbate_names'
        if 'hash' in _features:
            fingerprints['mpid'] = '$processed_data.calculation_info.mpid'
            fingerprints['miller'] = '$processed_data.calculation_info.miller'
            fingerprints['top'] = '$processed_data.calculation_info.top'
            fingerprints['coordination'] = '$processed_data.fp_final.coordination'
            fingerprints['nextnearestcoordination'] = '$processed_data.fp_init.nextnearestcoordination'
            fingerprints['neighborcoord'] = '$processed_data.fp_init.neighborcoord'
        # Some responses require specific queries. Here, we make sure that the correct
        # queries are defined
        _responses = dict.fromkeys(responses)
        if 'energy' in _responses:
            fingerprints['energy'] = '$results.energy'

        # Pull the data into parsed mongo documents (i.e., a dictionary of lists), `p_docs`
        _, p_docs = utils.get_docs(self.client, collection, fingerprints,
                                   adsorbates=None,
                                   calc_settings=None,
                                   vasp_settings=self.vasp_settings,
                                   energy_min=self.energy_min,
                                   energy_max=self.energy_max,
                                   f_max=self.f_max,
                                   ads_move_max=self.ads_move_max,
                                   bare_slab_move_max=self.bare_slab_move_max,
                                   slab_move_max=self.slab_move_max)
        if not p_docs.values()[0]:
            raise Exception('PullFeatures failed to find any matches. Please check your query settings.')

        # Initialize the outputs
        x = {'train': None, 'test': None, 'train+test': None}
        y = copy.deepcopy(x)
        split_p_docs = copy.deepcopy(x)

        # Preprocess the features
        pp = GASpyPreprocessor(p_docs, features)
        x['train+test'] = pp.transform(p_docs)
        # Pull out, stack, and numpy-array-ify the responses.
        # We might do real preprocessing to these one day. But not today.
        if len(responses) == 1:
            y['train+test'] = np.array(p_docs[responses[0]])
        elif len(responses) > 1:
            y['train+test'] = []
            for response in responses:
                y['train+test'].append(np.array(p_docs[response]))
            pdb.set_trace()
            y['train+test'] = np.concatenate(tuple(y['train+test']), axis=1)

        # Split the inputs and outputs. We also pull out indices for splitting so that we
        # can split `p_docs`
        x['train'], x['test'], y['train'], y['test'], indices_train, indices_test = \
                train_test_split(x['train+test'], y['train+test'], range(len(x['train+test'])),
                                 train_size=self.train_size, random_state=self.random_state)
        split_p_docs = {'train+test': p_docs,
                        'train': {fp: np.array(values)[indices_train] for fp, values in p_docs.iteritems()},
                        'test': {fp: np.array(values)[indices_test] for fp, values in p_docs.iteritems()},}

        return x, y, split_p_docs, pp
