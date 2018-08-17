'''
This submodule contains various classes that turn fingerprints
(formatted in dictionaries/JSONs/documents) into numeric vectors
so that they can be fed into regression pipelines.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pickle
import numpy as np
import mendeleev
from pymatgen.ext.matproj import MPRester
from gaspy.utils import read_rc


class InnerShellFingerprinter(object):
    '''
    This fingerprinter converts the "inner shell" atoms---i.e., the coordinated
    atoms---into a Nx4 array of numbers, where N is the maximum number of
    elements seen in any of the inner shells of the training set. Each 1x4
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
    def __init__(self, docs):
        '''
        Arg:
            docs    A list of dictionaries. Each dictionary must have the
                    'mpid', 'adsorbate', and 'energy' keys. Should probably
                    come from the `gaspy.gasdb.get_adsorption_docs` function.
        '''
        self.docs = docs
        self._calculate_dummy_fp()
        self._calculate_max_num_species()


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
        median_energies = list(self.median_adsorption_energies.values())
        avg_median_energy = np.average(median_energies)
        avg_atomic_num = np.average([self.mendeleev_data[element].atomic_number
                                     for element in elements])
        avg_electroneg = np.average([self.mendeleev_data[element].electronegativity(scale='pauling')
                                     for element in elements])
        dummy_count = 0
        self.dummy_fp = (avg_median_energy, avg_atomic_num, avg_electroneg, dummy_count)


    def _calculate_max_num_species(self):
        '''
        When populating "dummy fingerprints", we need to know how many of them to make.
        We set this number equal to the maximum number of elements present in any one
        alloy in the training set, and we find this number here.

        Resulting attributes:
            max_num_species     An integer for the maximum number of elements/species
                                present in any single mpid we are looking at. This is useful
                                for figuring out how many dummy features you need to add.
        '''
        mpids = set(doc['mpid'] for doc in self.docs)
        num_species_per_mpid = [len(self.compositions_by_mpid[mpid]) for mpid in mpids]
        self.max_num_species = max(num_species_per_mpid)


    def _get_compositions_by_mpid(self):
        '''
        We use the Materials Project's python API to find the composition of
        various materials given their MPIDs. This can take awhile though, so we also
        cache the results and modify the cache as necessary.

        Resulting attribute:
            compositions_by_mpid    A dictionary whose keys are MPIDs and whose values
                                    are lists of strings for each element that is present
                                    in the corresponding material. This object is cached
                                    and is therefore likely to have extra key:value pairings
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
        required_mpids = set(doc['mpid'] for doc in self.docs)
        unknown_mpids = set(required_mpids) - known_mpids

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
        mpids = set(doc['mpid'] for doc in self.docs)
        elements = []
        for mpid, composition in self.compositions_by_mpid.items():
            if mpid in mpids:
                elements.extend(composition)
        all_elements = set(elements)

        # Get the Mendeleev data for each element
        mendeleev_data = dict.fromkeys(all_elements)
        for element in all_elements:
            mendeleev_data[element] = getattr(mendeleev, element)
        self.mendeleev_data = mendeleev_data


    def _calculate_median_adsorption_energies(self):
        '''
        This method calculates the median adsorption energies on each monometallic bulk

        *Note*:
            This function uses only the data within the initial
            `docs` argument (fed to __init__) to calculate these values.

        Resulting attribute:
            median_adsorption_energies  A dictionary whose keys are the substrate elements
                                        found in `docs` and whose values are the median
                                        adsorption energy for that element (as per the
                                        doc['energy'] values in `docs`).
        '''
        # Prerequisite information
        docs, monometallic_elements = self._filter_out_alloys_from_docs()

        # Calculate the median adsorption energy for each element
        median_adsorption_energies = dict.fromkeys(monometallic_elements)
        for element in monometallic_elements:
            energies = [doc['energy'] for doc in docs
                        if self.compositions_by_mpid[doc['mpid']][0] == element]
            median = np.median(energies)

            # Sometimes our data is sparse and yields no energies to take medians on.
            # When this happens, just take the median of all elements.
            if np.isnan(median):
                energies = [doc['energy'] for doc in docs]
                median = np.median(energies)

            median_adsorption_energies[element] = median
        self.median_adsorption_energies = median_adsorption_energies


    def _filter_out_alloys_from_docs(self):
        '''
        Given a list of documents with an 'mpid' key, this method
        will return the list but with all documents removed that correspond
        with bulk materials that have more than one element. It will also
        return the composition key that it used to perform the filtering.

        Arg:
            docs    A list of dictionaries. Each dictionary must have the
                    'mpid' key.
        Outputs:
            filtered_docs       A list that's effectivly identical to the `docs`
                                object supplied to the user, but with alloys
                                removed.
            remaining_elements  A set of strings that contains the
                                monometallics that are present in `docs`
        '''
        filtered_docs = [doc for doc in self.docs if len(self.compositions_by_mpid[doc['mpid']]) == 1]
        remaining_elements = set(self.compositions_by_mpid[doc['mpid']][0] for doc in filtered_docs)
        return filtered_docs, remaining_elements


    def fingerprint_docs(self, docs, flatten=True):
        '''
        Convert a list of documents into a list of numerical fingerprints.

        Inputs:
            docs    A list of dictionaries that should have the keys 'mpid' and 'coordination'.
                    The value for 'mpid' should be in the form 'mpid-23' and the value
                    for 'coordination' should be in the form 'Cu-Cu-Cu'.
                    Should probably come from the `gaspy.gasdb.get_catalog_docs` function.
            flatten A boolean indicating whether or not you want to flatten the
                    output into a numpy vector, or keep it as a list of tuples.
                    You should probably flatten it if you plan to use it, but
                    you can not flatten it if you want to understand/view it better.
        Output:
            chem_fps    A list of tuples. The length of the list is equal to the number
                        of elements present in the coordination, and the length of the
                        tuples is 4. The first value in each tuple is the median
                        adsorption energy; the second value is the atomic number;
                        the third value is the Pauling electronegativity; and the last
                        number is the count of that element in the coordination site.
                        If `flatten == True`, then each fingerprint will be flattened into a
                        1-dimensional numpy array.
        '''
        chem_fps = [self.fingerprint_doc(doc, flatten=flatten) for doc in docs]
        return chem_fps


    def fingerprint_doc(self, doc, flatten=True):
        '''
        Convert a document into a numerical fingerprint.

        Inputs:
            doc     A dictionary that should have the keys 'mpid' and 'coordination'.
                    The value for 'mpid' should be in the form 'mpid-23' and the value
                    for 'coordination' should be in the form 'Cu-Cu-Cu'.
                    Should probably come from the `gaspy.gasdb.get_catalog_docs` function.
            flatten A boolean indicating whether or not you want to flatten the
                    output into a numpy vector, or keep it as a list of tuples.
                    You should probably flatten it if you plan to use it, but
                    you can not flatten it if you want to understand/view it better.
        Output:
            chem_fp A list of tuples. The length of the list is equal to the number
                    of elements present in the coordination, and the length of the
                    tuples is 4. The first value in each tuple is the median
                    adsorption energy; the second value is the atomic number;
                    the third value is the Pauling electronegativity; and the last
                    number is the count of that element in the coordination site.
                    If `flatten == True`, then this will be flattened into a
                    1-dimensional numpy array.
        '''
        chem_fp = []
        binding_atoms = doc['coordination'].split('-')

        # Sometimes there is no coordination. If this happens,
        # then hackily fix it
        if binding_atoms == ['']:
            binding_atoms = set()

        # Add and sort the elemental information for each element present
        for element in set(binding_atoms):
            try:
                element_data = self.mendeleev_data[element]
                energy = self.median_adsorption_energies[element]
                atomic_number = element_data.atomic_number
                electronegativity = element_data.electronegativity(scale='pauling')
                count = binding_atoms.count(element)
                chem_fp.append((energy, atomic_number, electronegativity, count))

            # Tell the user if they tried to predict outside of their test space
            except KeyError as error:
                import sys
                extra_message = ' because probably tried to fingerprint an element that was not in the training set'
                raise type(error)(str(error) + extra_message).with_traceback(sys.exc_info()[2])

        chem_fp = sorted(chem_fp)

        # Fill in the dummy fingerprints
        for _ in range(len(chem_fp), self.max_num_species):
            chem_fp.append(self.dummy_fp)

        # Format and flatten the data (if needed)
        chem_fp = np.array(chem_fp)
        if flatten:
            chem_fp = chem_fp.flatten()

        return chem_fp
