'''
This class contains different methods to pull and parse data from *.db files so that
other scripts may perform data analyses on them.

Note that this script uses the term "factors". Some people may call these "features" or
"independent variables".
'''
__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import sys
import pdb
import pickle
import numpy as np
from ase.db import connect
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from vasp.mongo import mongo_doc_atoms
sys.path.insert(0, '../')
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
        symbols = self._pull(['symbols'])['symbols']

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
        pulled_factors = ['coordination', 'adsorbate', 'mpid', 'miller']
        factors = ['coordination', 'adsorbate']
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
        # Pre-process the coordination
        p_data['coordination'], lb_coord = self._coord2coordcount(data['coordination'])

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


    def energy_fr_coordcount(self):
        '''
        Pull data according to the following motifs:
            coord_count     A vector of ordered integers. Each integer represents the number
                            of atoms of an element that are coordinated with the adsorbate.
                            For example:  a coordination site of Au-Ag-Ag could be represented
                            by [0, 0, 0, 1, 0, 2], where the zeros represent the coordination
                            counts for other elements (e.g., Al or Pt).
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
        pulled_factors = ['coordination', 'adsorbate', 'mpid', 'miller']
        factors = ['coordination']
        responses = ['energy']
        data = self._pull(pulled_factors+responses+['symbols'])
        # Initialize a second dictionary, `p_data`, that will be identical to the `data`
        # dictionary, except the values will be pre-processed data, not "raw" data.
        p_data = dict.fromkeys(factors+responses, None)

        # Pre-process the energy
        p_data['energy'] = np.array(data['energy'])
        # Pre-process the coordination
        p_data['coordination'], lb_coord = self._coord2coordcount(data['coordination'])

        # Stack the data to create the outputs
        x = self._stack(p_data, factors)
        y = self._stack(p_data, responses)

        # If specified, return the split data and the raw data
        if self.split:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=self.train_size,
                                                                random_state=self.random_state)
            return x, y, data, x_train, x_test, y_train, y_test, lb_coord
        # If we are not splitting the data, then simply returt x, y, and the raw data
        else:
            return x, y, data, lb_coord


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
        pulled_factors = ['coordination', 'nextnearestcoordination', 'adsorbate', 'mpid', 'miller']
        factors = ['coordination', 'nextnearestcoordination', 'adsorbate']
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
        # Pre-process the coordination counts
        p_data['coordination'], lb_coord = self._coord2coordcount(data['coordination'])
        # Pre-process the next nearest coordination counts
        p_data['nextnearestcoordination'], lb_nncoord = self._coord2coordcount(data['nextnearestcoordination'])

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
        pulled_factors = ['coordination', 'neighborcoord', 'bulkfwid',
                          'adsorbate', 'mpid', 'miller']
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
        # Determine the maximum coordination of each of the elements in the bulk, c_max.
        # This will also be used to calculate the GCNs. Note that `cmax` will be a nested
        # dictionary. The highest level keys are the fwids of the bulks; the second level
        # keys (which are the first-level values) are the unique elements of that bulk;
        # and the second-level values are naive coordination numbers (i.e., they treat
        # all elements the same).
        pkl_name = './pkls/cmax.pkl'
        try:
            # Calculating `cmax` takes a solid amount of time. Let's see if an old,
            # pickled version of it is lying around before we try calculating it
            # all over again.
            with open(pkl_name, 'rb') as fname:
                cmax = pickle.load(fname)
        except IOError:
            with utils.get_aux_db() as aux_db:
                # `bulks` is a dictionary of all of the bulks we've relaxed, where
                # the key is the fwid and the value is the ase.Atoms object.
                bulks = {fwid: mongo_doc_atoms(aux_db.find({'fwid': fwid})[0])
                         for fwid in np.unique(data['bulkfwid'])}
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
                        neighbor_coords[atom.symbol].append(np.sum(lb_coord.transform(neighbor_atoms)))
                    except Exception as ex:
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
            # Save our results to a pickle so we don't have to do this again.
            with open(pkl_name, 'wb') as fname:
                pickle.dump(cmax, fname)

        # Pre-process the GCNs. But first we initialize and then turn the GCN
        # array into a float array, since we'll be adding fractional coordination
        # numbers
        p_data['gcn'], __lb = self._coord2coordcount(['']*len(data['energy']))
        p_data['gcn'] = p_data['gcn'].astype(float)
        for i, coord in enumerate(data['coordination']):
            # `data` contains the coordinations as strings,
            # e.g., "[u'W:N-N', u'W:N-N-N']". This indexing removes the outer shell
            # to leave a W:N-N', u'W:N-N-N
            neighbor_coords = data['neighborcoord'][i][3:-2]
            # Split `neighbor_coords` from one string to a list, where each
            # element corresponds to a different neighbor. Now the form will be
            # ['W:N-N', 'W:N-N-N']
            neighbor_coords = neighbor_coords.split('\', u\'')
            # Take out the pre-labels, leaving a ['N-N', 'N-N-N']. The indexing
            # of this list is identical to the order-of-appearance within `coord`
            neighbor_coords = [neighbor_coord.split(':')[1] for neighbor_coord in neighbor_coords]
            # Calculate and assign the gcn contribution for each neighbor
            for j, neighbor in enumerate(coord.split('-')):
                # The classical coordination number of the neighbor (excluding adsorbate)
                neighbor_cn = len(neighbor_coords[j].split('-'))
                try:
                    _cmax = cmax[data['bulkfwid'][i]][neighbor]
                    # The index of `neighbor`'s element within the coordcount vector
                    index = self._coord2coordcount([neighbor])[0].tolist()[0].index(1)
                    try:
                        # Add the gcn contribution from this neighor.
                        p_data['gcn'][i][index] += neighbor_cn/_cmax
                    # If _cmax somehow turns out to be zero, then add a gcn of
                    # 1 (if the neighbor is coordinated at all)
                    except ZeroDivisionError:
                        if neighbor_cn:
                            p_data['gcn'][i][index] += 1
                # If the `coord` ends up containing an element that's not
                # in the bulk (which only happens on edge cases where
                # PyMatGen fails us), then simply set the neighbor_cn to 1
                # (if the neighbor is coordinated at all).
                except KeyError:
                    if neighbor_cn:
                        p_data['gcn'][i][index] += 1
            try:
                if not i % 100:
                    print("Pulling out data point #%s for GCN" % i)
            except ZeroDivisionError:
                pass

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


    def energy_fr_gcn(self):
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
        pulled_factors = ['coordination', 'neighborcoord', 'bulkfwid',
                          'adsorbate', 'mpid', 'miller']
        factors = ['gcn']
        responses = ['energy']
        data = self._pull(pulled_factors+responses+['symbols'])
        # Initialize a second dictionary, `p_data`, that will be identical to the `data`
        # dictionary, except the values will be pre-processed data, not "raw" data.
        p_data = dict.fromkeys(factors+responses, None)

        # Pre-process the energy
        p_data['energy'] = np.array(data['energy'])

        # Create a binarizer for the elements we are looking at so that we can use
        # it to calculate the GCNs
        lb_coord = preprocessing.LabelBinarizer()
        lb_coord.fit(np.unique([item for sublist in data['symbols'] for item in sublist
                                if item != 'C' and item != 'O']))
        # Determine the maximum coordination of each of the elements in the bulk, c_max.
        # This will also be used to calculate the GCNs. Note that `cmax` will be a nested
        # dictionary. The highest level keys are the fwids of the bulks; the second level
        # keys (which are the first-level values) are the unique elements of that bulk;
        # and the second-level values are naive coordination numbers (i.e., they treat
        # all elements the same).
        pkl_name = './pkls/cmax.pkl'
        try:
            # Calculating `cmax` takes a solid amount of time. Let's see if an old,
            # pickled version of it is lying around before we try calculating it
            # all over again.
            with open(pkl_name, 'rb') as fname:
                cmax = pickle.load(fname)
        except IOError:
            with utils.get_aux_db() as aux_db:
                # `bulks` is a dictionary of all of the bulks we've relaxed, where
                # the key is the fwid and the value is the ase.Atoms object.
                bulks = {fwid: mongo_doc_atoms(aux_db.find({'fwid': fwid})[0])
                         for fwid in np.unique(data['bulkfwid'])}
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
                        neighbor_coords[atom.symbol].append(np.sum(lb_coord.transform(neighbor_atoms)))
                    except Exception as ex:
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
            # Save our results to a pickle so we don't have to do this again.
            with open(pkl_name, 'wb') as fname:
                pickle.dump(cmax, fname)

        # Pre-process the GCNs. But first we initialize and then turn the GCN
        # array into a float array, since we'll be adding fractional coordination
        # numbers
        p_data['gcn'], __lb = self._coord2coordcount(['']*len(data['energy']))
        p_data['gcn'] = p_data['gcn'].astype(float)
        for i, coord in enumerate(data['coordination']):
            # `data` contains the coordinations as strings,
            # e.g., "[u'W:N-N', u'W:N-N-N']". This indexing removes the outer shell
            # to leave a W:N-N', u'W:N-N-N
            neighbor_coords = data['neighborcoord'][i][3:-2]
            # Split `neighbor_coords` from one string to a list, where each
            # element corresponds to a different neighbor. Now the form will be
            # ['W:N-N', 'W:N-N-N']
            neighbor_coords = neighbor_coords.split('\', u\'')
            # Take out the pre-labels, leaving a ['N-N', 'N-N-N']. The indexing
            # of this list is identical to the order-of-appearance within `coord`
            neighbor_coords = [neighbor_coord.split(':')[1] for neighbor_coord in neighbor_coords]
            # Calculate and assign the gcn contribution for each neighbor
            for j, neighbor in enumerate(coord.split('-')):
                # The classical coordination number of the neighbor (excluding adsorbate)
                neighbor_cn = len(neighbor_coords[j].split('-'))
                try:
                    _cmax = cmax[data['bulkfwid'][i]][neighbor]
                    # The index of `neighbor`'s element within the coordcount vector
                    index = self._coord2coordcount([neighbor])[0].tolist()[0].index(1)
                    try:
                        # Add the gcn contribution from this neighor.
                        p_data['gcn'][i][index] += neighbor_cn/_cmax
                    # If _cmax somehow turns out to be zero, then add a gcn of
                    # 1 (if the neighbor is coordinated at all)
                    except ZeroDivisionError:
                        if neighbor_cn:
                            p_data['gcn'][i][index] += 1
                # If the `coord` ends up containing an element that's not
                # in the bulk (which only happens on edge cases where
                # PyMatGen fails us), then simply set the neighbor_cn to 1
                # (if the neighbor is coordinated at all).
                except KeyError:
                    if neighbor_cn:
                        p_data['gcn'][i][index] += 1
            try:
                if not i % 100:
                    print("Pulling out data point #%s for GCN" % i)
            except ZeroDivisionError:
                pass

        # Stack the data to create the outputs
        x = self._stack(p_data, factors)
        y = self._stack(p_data, responses)

        # If specified, return the split data and the raw data
        if self.split:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=self.train_size,
                                                                random_state=self.random_state)
            return x, y, data, x_train, x_test, y_train, y_test, lb_coord
        # If we are not splitting the data, then simply returt x, y, and the raw data
        else:
            return x, y, data, lb_coord
