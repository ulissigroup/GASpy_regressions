'''
This class contains different methods to pull and parse data from *.db files so that
other scripts may perform data analyses on them.

Note that this script uses the term "factors". Some people may call these "features" or
"independent variables".
'''
__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import pdb
import numpy as np
import pickle
from ase.db import connect
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from vasp.mongo import mongo_doc_atoms
from gaspy import utils


class GASPull(object):
    def __init__(self, db_loc, vasp_settings,
                 split=False,
                 energy_min=-4, energy_max=4, slab_move_max=1.5, ads_move_max=1.5, fmax_max=0.5,
                 train_size=0.75, random_state=42):
        '''
        Connect to the database and pull out the pertinent rows.
        INPUTS:
        db_loc          The full name of the path to the *.db file. Do not include the *.db file
                        the path
        vasp_settings   A string of vasp settings. Use the vasp_settings_to_str function in GAspy
        split           Boolean that is true if you want to split the data into a training and
                        test sets
        energy_min      The minimum adsorption energy to pull from the Local DB (eV)
        energy_max      The maximum adsorption energy to pull from the Local DB (eV)
        ads_move_max    The maximum distance that an adsorbate atom may move (angstrom)
        slab_move_max   The maximum distance that a slab atom may move (angstrom)
        fmax_max        The upper limit on the maximum force on an atom in the system
        '''
        # Pass along various parameters to use later
        self.split = split
        self.train_size = train_size
        self.random_state = random_state

        # Update PYTHONPATH so we can connect to the Local database, and then pull from it
        with connect(db_loc+'/adsorption_energy_database.db') as db:
            # A list of ase-db rows are stored in self.rows for later use
            self.rows = [row for row in db.select()
                         if all([row[key] == vasp_settings[key] for key in vasp_settings])
                         and energy_min < row.energy < energy_max
                         and row.max_surface_movement < slab_move_max
                         and row.max_adsorbate_movement < ads_move_max
                         and row.fmax < fmax_max]
            # If we did not pull anything from the database, then stop the script and alert the user
            if len(self.rows) == 0:
                raise Exception('DATABASE ERROR:  Could not find any database rows to match input settings. Please verify db_loc, vasp_settings, or whether or not the database actually has the data you are looking for.')


    def _pull(self, variables):
        '''
        This function pulls data from the database.
        Input:
            `variables`, is a list of strings that correspond to the ase-db row attribute
            that you want to pull. These strings must correspond to the row attributes,
            i.e., row[string] must return a value.
        Output:
            `data` is a dictionary whose keys are the variables names and whose values are
            lists of the data
        '''
        data = dict.fromkeys(variables, None)
        for variable in variables:
            data[variable] = [row[variable] for row in self.rows]
        return data


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


    def energy_fr_coordcount_ads(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            ads             A vector of binaries that indicate the type of adsorbate.
        Outputs:
            data        A dictionary containing the data we pulled from the Local database.
                        This object is taken raw from the _pull method.
            x           A stacked array containing all of the data for all of the factors
            y           A stacked array containing all of the data for all of the responses
            x_train     A subset of `x` intended to use as a training set
            y_train     A subset of `y` intended to use as a training set
            x_test      A subset of `x` intended to use as a validation set
            y_test      A subset of `y` intended to use as a validation set
            lb_ads      The label binarizer used to binarize the adsorbate
            lb_coord    The label binarizer used to binarize the coordination vector
        '''
        # Establish the variables and pull the data from the Local database
        factors = ['coordination', 'adsorbate']
        responses = ['energy']
        data = self._pull(factors+responses+['symbols'])
        # Initialize a second dictionary, `p_data`, that will be identical to the `data`
        # dictionary, except the values will be pre-processed data, not "raw" data.
        p_data = dict.fromkeys(factors+responses, None)

        # Pre-process the energy
        p_data['energy'] = np.array(data['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(data['adsorbate'])
        lb_ads = preprocessing.LabelBinarizer()
        lb_ads.fit(ads)
        p_data['adsorbate'] = lb_ads.transform(data['adsorbate'])
        # Pre-process the coordination
        lb_coord = preprocessing.LabelBinarizer()
        lb_coord.fit(np.unique([item for sublist in data['symbols'] for item in sublist
                                if item != 'C' and item != 'O']))
                                # We filter out C & O because Alamo cries if we include
                                # symbols that do not actually end up as part of the
                                # coordination. We can put these back in if we start dealing
                                # with carbides or oxides, respectively.
        p_data['coordination'] = np.array([np.sum(lb_coord.transform(coord.split('-')), axis=0)
                                           for coord in data['coordination']])

        # Stack the data to create the outputs
        x = self._stack(p_data, factors)
        y = self._stack(p_data, responses)

        # If specified, return the split data and the raw data
        if self.split:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=self.train_size,
                                                                random_state=self.random_state)
            return x, y, data, x_train, x_test, y_train, y_test, lb_ads, lb_coord
        # If we are not splitting the data, then simply returt x, y, and the raw data
        else:
            return x, y, data, lb_ads, lb_coord


    def energy_fr_coordcount_neighborcount_ads(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
            neighbor_count  The same as coord_count, except we cound the atoms that are
                            coordinated with the binding atoms' neighbors (i.e., the total
                            coordination of the coordinated atoms).
            ads             A vector of binaries that indicate the type of adsorbate.
        Outputs:
            data        A dictionary containing the data we pulled from the Local database.
                        This object is taken raw from the _pull method.
            x           A stacked array containing all of the data for all of the factors
            y           A stacked array containing all of the data for all of the responses
            x_train     A subset of `x` intended to use as a training set
            y_train     A subset of `y` intended to use as a training set
            x_test      A subset of `x` intended to use as a validation set
            y_test      A subset of `y` intended to use as a validation set
            lb_ads      The label binarizer used to binarize the adsorbate
            lb_coord    The label binarizer used to binarize the coordination vector
        '''
        # Establish the variables and pull the data from the Local database
        factors = ['coordination', 'nextnearestcoordination', 'adsorbate']
        responses = ['energy']
        data = self._pull(factors+responses+['symbols'])
        # Initialize a second dictionary, `p_data`, that will be identical to the `data`
        # dictionary, except the values will be pre-processed data, not "raw" data.
        p_data = dict.fromkeys(factors+responses, None)

        # Pre-process the energy
        p_data['energy'] = np.array(data['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(data['adsorbate'])
        lb_ads = preprocessing.LabelBinarizer()
        lb_ads.fit(ads)
        p_data['adsorbate'] = lb_ads.transform(data['adsorbate'])
        # Pre-process the coordination counts
        lb_coord = preprocessing.LabelBinarizer()
        lb_coord.fit(np.unique([item for sublist in data['symbols'] for item in sublist
                                if item != 'C' and item != 'O']))
                                # We filter out C & O because Alamo cries if we include
                                # symbols that do not actually end up as part of the
                                # coordination. We can put these back in if we start dealing
                                # with carbides or oxides, respectively.
        p_data['coordination'] = np.array([np.sum(lb_coord.transform(coord.split('-')), axis=0)
                                           for coord in data['coordination']])
        # Pre-process the secondary coordination counts. Note that we use the same binarizer
        # that we used for coordination.
        p_data['nextnearestcoordination'] = \
                np.array([np.sum(lb_coord.transform(ncoord.split('-')), axis=0)
                          for ncoord in data['nextnearestcoordination']])

        # Stack the data to create the outputs
        x = self._stack(p_data, factors)
        y = self._stack(p_data, responses)

        # If specified, return the split data and the raw data
        if self.split:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=self.train_size,
                                                                random_state=self.random_state)
            return x, y, data, x_train, x_test, y_train, y_test, lb_ads, lb_coord
        # If we are not splitting the data, then simply returt x, y, and the raw data
        else:
            return x, y, data, lb_ads, lb_coord


    def energy_fr_gcn_ads(self):
        # TODO:  Find a way to determine robustly the true maximal coordination
        # of each atom, instead of assuming that each element in the bulk has the
        # same maximal coordination.
        '''
        Pull data according to the following motifs:
            gcn     Generalized Coordination Number for an adsorbate on an alloy.
                    Specifically, it is a horizontally-stacked, ordered, 2-D array of
                    integers. The i'th vector in the array represents the cumulative
                    coordination of all i-element atoms that are coordinated with
                    the adsorbate directly. Each j'th element in each vector
                    represents the cumulative coordination of each j'th neareast-
                    neighbor that is also coordinated with an i'th neighbor.
                    ...I'll write it down in better detail later, and probably
                    somewhere else.
            ads     A vector of binaries that indicate the type of adsorbate.
        Outputs:
            data        A dictionary containing the data we pulled from the Local database.
                        This object is taken raw from the _pull method.
            x           A stacked array containing all of the data for all of the factors
            y           A stacked array containing all of the data for all of the responses
            x_train     A subset of `x` intended to use as a training set
            y_train     A subset of `y` intended to use as a training set
            x_test      A subset of `x` intended to use as a validation set
            y_test      A subset of `y` intended to use as a validation set
            lb_ads      The label binarizer used to binarize the adsorbate
            lb_coord    The label binarizer used to binarize the coordination vector
        '''
        # Establish the variables and pull the data from the Local database
        pulled_factors = ['coordination', 'nextnearestcoordination', 'bulkfwid',
                          'adsorbate']
        factors = ['gcn', 'adsorbate']
        responses = ['energy']
        data = self._pull(pulled_factors+responses+['symbols'])
        # Initialize a second dictionary, `p_data`, that will be identical to the `data`
        # dictionary, except the values will be pre-processed data, not "raw" data.
        p_data = dict.fromkeys(factors+responses, None)

        # Pre-process the energy
        p_data['energy'] = np.array(data['energy'])
        # Pre-process the adsorbate identity via binarizer
        ads = np.unique(data['adsorbate'])
        lb_ads = preprocessing.LabelBinarizer()
        lb_ads.fit(ads)
        p_data['adsorbate'] = lb_ads.transform(data['adsorbate'])

        # Create a binarizer for the elements we are looking at so that we can use
        # it to calculate the GCNs
        lb_coord = preprocessing.LabelBinarizer()
        lb_coord.fit(np.unique([item for sublist in data['symbols'] for item in sublist
                                if item != 'C' and item != 'O']))
                                # We filter out C & O because Alamo cries if we include
                                # symbols that do not actually end up as part of the
                                # coordination. We can put these back in if we start dealing
                                # with carbides or oxides, respectively.
        # Determine the maximum coordination of each of the elements in the bulk, c_max.
        # This will also be used to calculate the GCNs
        try:
            with open('./pkls/cmax.pkl', 'rb') as fname:
                cmax = pickle.load(fname)
        except IOError:
            with utils.get_aux_db() as aux_db:
                bulks = {fwid: mongo_doc_atoms(aux_db.find({'fwid': fwid})[0])
                         for fwid in np.unique(data['bulkfwid'])}
            cmax = dict.fromkeys(bulks)
            for fwid, bulk in bulks.iteritems():
                struct = AseAtomsAdaptor.get_structure(bulk)
                vcf = VoronoiCoordFinder(struct, allow_pathological=True)
                bulk_symbols = np.unique(bulk.get_chemical_symbols())
                counts = dict.fromkeys(bulk_symbols, [])
                for i, atom in enumerate(bulk):
                    try:
                        neighbor_sites = vcf.get_coordinated_sites(i, 0.8)
                        neighbor_atoms = [neighbor_site.species_string
                                          for neighbor_site in neighbor_sites]
                        counts[atom.symbol].append(np.sum(lb_coord.transform(neighbor_atoms),
                                                          axis=0))
                    except Exception as ex:
                        template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
                        message = template.format(type(ex).__name__, ex.args)
                        print(message)
                        print('We might have gotten a Qhull error at fwid=%s, atom #%s (%s)' \
                              % (fwid, i, atom.symbol))
                        counts[atom.symbol].append([0]*len(lb_coord.transform([''])))
                cmax[fwid] = {symbol: np.maximum.reduce(count)
                              for symbol, count in counts.iteritems()}
            with open('./pkls/cmax.pkl', 'wb') as fname:
                pickle.dump(cmax, fname)

        ## Pre-process the GCNs.
        #p_data['coordination'] = np.array([np.sum(lb_coord.transform(coord.split('-')), axis=0)
                                           #for coord in data['coordination']])
        #p_data['nextnearestcoordination'] = \
                #np.array([np.sum(lb_coord.transform(ncoord.split('-')), axis=0)
                          #for ncoord in data['nextnearestcoordination']])
#
        ## Stack the data to create the outputs
        #x = self._stack(p_data, factors)
        #y = self._stack(p_data, responses)
#
        ## If specified, return the split data and the raw data
        #if self.split:
            #x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                #train_size=self.train_size,
                                                                #random_state=self.random_state)
            #return x, y, data, x_train, x_test, y_train, y_test, lb_ads, lb_coord
        ## If we are not splitting the data, then simply returt x, y, and the raw data
        #else:
            #return x, y, data, lb_ads, lb_coord
