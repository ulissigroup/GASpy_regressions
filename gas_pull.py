'''
This class contains different methods to pull and parse data from *.db files so that
other scripts may perform data analyses on them.

Note that this script uses the term "factors". Some people may call these "features" or
"independent variables".
'''
__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import pdb
import sys
import numpy as np
from ase.db import connect
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class GASPullByMotifs(object):
    def __init__(self, db_loc, vasp_settings,
                 split=False,
                 energy_min=-4, energy_max=4, slab_move_max=1.5, fmax_max=0.5,
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
        slab_move_max   The maximum distance that a slab atom may move (angstrom)
        fmax_max        The upper limit on the maximum force on an atom in the system
        '''
        # Pass along various parameters to use later
        self.split = split
        self.train_size = train_size
        self.random_state = random_state

        # Update PYTHONPATH so we can connect to the Local database, and then pull from it
        sys.path.append(db_loc)
        db = connect(db_loc+'/adsorption_energy_database.db')
        # A list of ase-db rows are stored in self.rows for later use
        self.rows = [row for row in db.select()
                     if all([row[key] == vasp_settings[key] for key in vasp_settings])
                     and energy_min < row.energy < energy_max
                     and row.max_surface_movement < slab_move_max
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


    def coordcount_ads_to_energy(self):
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
        elif not self.split:
            return x, y, data, lb_ads, lb_coord
