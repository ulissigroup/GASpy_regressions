'''
This submodule contains various functions that turn fingerprints
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
    def __init__(self, docs):
        '''
        Arg:
            docs    A list of dictionaries. Each dictionary must have the
                    'mpid', 'adsorbate', and 'energy' keys. Should probably
                    come from the `gaspy.gasdb.get_adsorption_docs` function.
        '''
        self.docs = docs
        self._calculate_dummy_fp()


    def _calculate_dummy_fp(self):
        '''
        This method establishes a "dummy" value for the `chemfp0` type of feature.
        This dummy value is useful when using variable number of features;
        reference Davie et al (Kriging atomic properties with a variable number of inputs,
        J Chem Phys 2016). The out-of-bounds feature we choose is the atomic count.

        Resulting attributes:
            dummy_fp            A tuple that represents a single chemfp0 fingerprint,
                                but has the "dummy" values
            max_num_species     An integer for the maximum number of elements/species
                                present in any single mpid we are looking at. This is useful
                                for figuring out how many dummy features you need to add.
        '''
        # Prerequisite calculations
        self._get_compositions_by_mpid()
        self._get_mendeleev_data()
        self._calculate_median_adsorption_energies()
        mpids = set(doc['mpid'] for doc in self.docs)
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

        # Calculate `max_num_species`
        num_species = [len(self.compositions_by_mpid[mpid]) for mpid in mpids]
        self.max_num_species = max(num_species)


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

        *Notes*:
            This function uses only the data within the `docs` argument to calculate these values.
            This function also assumes that you have only one adsorbate---i.e., no co-adsorption

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


    def transform(self, docs):
        return 'pending'
