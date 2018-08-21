'''
This submodule contains various classes that turn fingerprints
(formatted in dictionaries/JSONs/documents) into numeric vectors
so that they can be fed into regression pipelines.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import pickle
import numpy as np
import mendeleev
from pymatgen.ext.matproj import MPRester
from gaspy.utils import read_rc
from gaspy.gasdb import get_adsorption_docs, get_catalog_docs


class InnerShellFingerprinter(object):
    '''
    This fingerprinter converts the "inner shell" atoms---i.e., the coordinated
    atoms---into a Nx4 array of numbers, where N is the maximum number of
    elements seen in any of the inner shells all sites in the catalog. Each 1x4
    vector corresponds to one of the elements present in this inner shell. The
    numbers in the 1x4 vectors are the element's median adsorption energy, its
    atomic number, its Pauling electronegativity, and the number of those
    elements that show up in the coordination. We also sort the 1x4 vectors
    such that the first 1x4 vector that shows up is the one with the lowest
    median adsorption energy.

    We also use "dummy" vectors to address the fact that we will have a
    variable number of features/elements present in the inner shell; reference
    Davie et al (Kriging atomic properties with a variable number of inputs, J
    Chem Phys 2016). The out-of-bounds feature we choose is the atomic count.
    '''
    def __init__(self, adsorbate):
        '''
        Arg:
            adsorbate   A string indicating which adsorbate you want to
                        make fingerprints for
        '''
        self.adsorbate = adsorbate

        # Get the data that we need to calculate the prerequisite information
        self.adsorption_docs = get_adsorption_docs(adsorbates=[adsorbate])
        self.catalog_docs = get_catalog_docs()

        # Calculate the information we need to make a fingerprint
        self._calculate_dummy_fp()
        self._get_max_num_species()

        # Delete some data to save memory
        del self.adsorption_docs
        del self.catalog_docs


    def _calculate_dummy_fp(self):
        '''
        This method establishes a "dummy" value for the 1x4 vector of information.
        This dummy value is useful when using variable number of features;
        reference Davie et al (Kriging atomic properties with a variable number of inputs,
        J Chem Phys 2016). The out-of-bounds feature we choose is the atomic count.

        Resulting attributes:
            dummy_fp            A tuple that represents a single chemfp0 fingerprint,
                                but has the "dummy" values
        '''
        # Prerequisite calculations
        self._get_compositions_by_mpid()
        self._get_mendeleev_data()
        self._calculate_median_adsorption_energies()
        elements = set(self.median_adsorption_energies.keys())

        # Calculate `dummy_fp`
        avg_median_energy = np.average(list(self.median_adsorption_energies.values()))
        avg_atomic_num = np.average([self.mendeleev_data[element].atomic_number
                                     for element in elements])
        avg_electroneg = np.average([self.mendeleev_data[element].electronegativity(scale='pauling')
                                     for element in elements])
        dummy_count = 0
        self.dummy_fp = (avg_median_energy, avg_atomic_num, avg_electroneg, dummy_count)


    def _get_compositions_by_mpid(self):
        '''
        We use the Materials Project's python API to find the composition of
        various materials given their MPIDs. This can take awhile though, so we also
        cache the results and modify the cache as necessary.

        Resulting attribute:
            compositions_by_mpid    A dictionary whose keys are MPIDs and whose values
                                    are lists of strings for each element that is present
                                    in the corresponding material. This object is cached
                                    and therefore may have extra key:value pairings
                                    that you may not need.
        '''
        # Find the current cache of compositions. If it's not there, then initialize it as an empty dict
        try:
            with open('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl', 'rb') as file_handle:
                compositions_by_mpid = pickle.load(file_handle)
        except FileNotFoundError:
            compositions_by_mpid = {}

        # Figure out which compositions we still need to figure out
        known_mpids = set(compositions_by_mpid.keys())
        required_mpids = set(doc['mpid'] for doc in self.adsorption_docs) | \
                         set(doc['mpid'] for doc in self.catalog_docs)
        unknown_mpids = required_mpids - known_mpids

        # If necessary, find the unknown compositions and save them to the cache
        if unknown_mpids:
            with MPRester(read_rc('matproj_api_key')) as rester:
                for mpid in unknown_mpids:
                    entry = rester.get_entry_by_material_id({'task_ids': mpid})
                    composition = entry.as_dict()['composition']
                    compositions_by_mpid[mpid] = list(composition.keys())
            with open('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl', 'wb') as file_handle:
                pickle.dump(compositions_by_mpid, file_handle)

        self.compositions_by_mpid = compositions_by_mpid


    def _get_mendeleev_data(self):
        '''
        This method will get all of the Mendeleev data for the substrates
        included in a set of aggregated Mongo documents.

        Resulting attribute:
            mendeleev_data  A dictionary whose keys are the elements present in
                            `docs` and whose values are the Mendeleev data
        '''
        # Find all of the elements we want to get data for
        elements = []
        for mpid, composition in self.compositions_by_mpid.items():
            elements.extend(composition)
        elements = set(elements)

        # Get the Mendeleev data for each element
        mendeleev_data = dict.fromkeys(elements)
        for element in mendeleev_data:
            mendeleev_data[element] = getattr(mendeleev, element)
        self.mendeleev_data = mendeleev_data


    def _calculate_median_adsorption_energies(self):
        '''
        This method calculates the median adsorption energies on each monometallic bulk.

        Resulting attribute:
            median_adsorption_energies  A dictionary whose keys are the substrate elements
                                        found in `docs` and whose values are the median
                                        adsorption energy for that element (as per the
                                        doc['energy'] values in `docs`).
        '''
        # Figure out the elements we need to calculate energies for
        elements = []
        for doc in self.adsorption_docs + self.catalog_docs:
            composition = self.compositions_by_mpid[doc['mpid']]
            elements.extend(composition)
        elements = set(elements)

        # Calculate the median adsorption energy for each element
        median_adsorption_energies = dict.fromkeys(elements)
        for element in median_adsorption_energies:
            energies = []
            for doc in self.adsorption_docs:
                composition = self.compositions_by_mpid[doc['mpid']]
                if len(composition) == 1 and composition[0] == element:
                    energies.append(doc['energy'])
            median = np.median(energies)

            # Sometimes our data is sparse and yields no energies to take medians on.
            # When this happens, just take the median of all elements.
            if np.isnan(median):
                energies = [doc['energy'] for doc in self.adsorption_docs]
                median = np.median(energies)
                message = 'We do not have any energy data for %s, so we set its median adsorption energy as the median of all energies' % element
                warnings.warn(message, RuntimeWarning)

            median_adsorption_energies[element] = median
        self.median_adsorption_energies = median_adsorption_energies


    def _get_max_num_species(self):
        '''
        When populating "dummy fingerprints", we need to know how many of them to make.
        We set this number equal to the maximum number of elements present in any one
        alloy in the catalog, and we find this number here.

        Resulting attributes:
            max_num_species     An integer for the maximum number of elements/species
                                present in any single mpid we are looking at. This is useful
                                for figuring out how many dummy features you need to add.
        '''
        mpids = set(doc['mpid'] for doc in self.catalog_docs)
        num_species_per_mpid = [len(self.compositions_by_mpid[mpid]) for mpid in mpids]
        self.max_num_species = max(num_species_per_mpid)


    def fingerprint_docs(self, docs):
        '''
        Convert a list of documents into a list of numerical fingerprints.

        Inputs:
            docs    A list of dictionaries that should have the keys 'mpid' and 'coordination'.
                    The value for 'mpid' should be in the form 'mpid-23' and the value
                    for 'coordination' should be in the form 'Cu-Cu-Cu'.
                    Should probably come from the `gaspy.gasdb.get_catalog_docs` function.
        Output:
            chem_fps    A list of numpy.array objects. Each numpy array is a
                        numerical representation of each document that you gave this
                        method, as per the docstring of this class. Note that
                        the array is actually a flattened, 1-dimensional object.
        '''
        chem_fps = [self.fingerprint_doc(doc) for doc in docs]
        return chem_fps


    def fingerprint_doc(self, doc):
        '''
        Convert a document into a numerical fingerprint.

        Inputs:
            doc     A dictionary that should have the keys 'mpid' and 'coordination'.
                    The value for 'mpid' should be in the form 'mpid-23' and the value
                    for 'coordination' should be in the form 'Cu-Cu-Cu'.
                    Should probably come from the `gaspy.gasdb.get_catalog_docs` function.
        Output:
            chem_fp     A numpy.array object that is a numerical representation the
                        document that you gave this method, as per the docstring of
                        this class. Note that the array is actually a flattened,
                        1-dimensional object.
        '''
        chem_fp = []
        binding_atoms = doc['coordination'].split('-')

        # Sometimes there is no coordination. If this happens, then hackily fix it
        if binding_atoms == ['']:
            binding_atoms = []

        # Add and sort the elemental information for each element present
        for element in set(binding_atoms):
            energy = self.median_adsorption_energies[element]
            element_data = self.mendeleev_data[element]
            atomic_number = element_data.atomic_number
            electronegativity = element_data.electronegativity(scale='pauling')
            count = binding_atoms.count(element)
            chem_fp.append((energy, atomic_number, electronegativity, count))
        chem_fp = sorted(chem_fp)

        # Fill in the dummy fingerprints
        for _ in range(len(chem_fp), self.max_num_species):
            chem_fp.append(self.dummy_fp)

        return np.array(chem_fp).flatten()
