'''
Tests for the `fingerprints` submodule.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..fingerprints import InnerShellFingerprinter

# Things we need to do the tests
import os
import pytest
import pickle
import mendeleev
from pymatgen.ext.matproj import MPRester
from gaspy.gasdb import get_adsorption_docs, get_catalog_docs
from gaspy.utils import read_rc

REGRESSION_BASELINES_LOCATION = '/home/GASpy/GASpy_regressions/gaspy_regress/tests/regression_baselines/fingerprints/'


# We should probably add more test cases here...
@pytest.fixture(params=[get_adsorption_docs()])
def inner_shell_fingerprinter(request):
    try:
        # Remove the cache to make sure that the new instance of the fingerprinter
        # is the one that's actually making the cache correctly
        os.remove('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl')
    except OSError:
        pass

    return InnerShellFingerprinter(request.param)


class TestInnerShellFingerprinter(object):
    @pytest.mark.baseline
    def test_to_create_dummy_fp(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'dummy_fp_inner_shell.pkl', 'wb') as file_handle:
            pickle.dump(inner_shell_fingerprinter.dummy_fp, file_handle)


    def test__calculate_dummy_fp(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'dummy_fp_inner_shell.pkl', 'rb') as file_handle:
            expected_dummy_fp = pickle.load(file_handle)
        assert inner_shell_fingerprinter.dummy_fp == expected_dummy_fp


    @pytest.mark.baseline
    def test_to_create_max_num_species(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'max_species_inner_shell.pkl', 'wb') as file_handle:
            pickle.dump(inner_shell_fingerprinter.max_num_species, file_handle)


    def test__calculate_max_num_species(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'max_species_inner_shell.pkl', 'rb') as file_handle:
            expected_max_num_species = pickle.load(file_handle)
        assert inner_shell_fingerprinter.max_num_species == expected_max_num_species


    def test__get_compositions_by_mpid(self, inner_shell_fingerprinter):
        '''
        3-part test:
            Test that the method saved the cache
            Test that all the MPIDs we need are there
            Test that all the data in the cache is correct
        '''
        # Make sure that we saved the object correctly
        with open('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl', 'rb') as file_handle:
            saved_compositions_by_mpid = pickle.load(file_handle)
        compositions_by_mpid = inner_shell_fingerprinter.compositions_by_mpid
        assert compositions_by_mpid == saved_compositions_by_mpid

        # Make sure that all of the required MPIDs are in the new object
        required_mpids = set(doc['mpid'] for doc in inner_shell_fingerprinter.docs)
        known_mpids = set(inner_shell_fingerprinter.compositions_by_mpid.keys())
        assert required_mpids.issubset(known_mpids)

        # Make sure that the compositions we do have are correct
        with MPRester(read_rc('matproj_api_key')) as rester:
            for mpid in known_mpids:
                entry = rester.get_entry_by_material_id({'task_ids': mpid})
                expected_composition = list(entry.as_dict()['composition'].keys())
                assert compositions_by_mpid[mpid] == expected_composition


    def test__get_mendeleev_data(self, inner_shell_fingerprinter):
        mendeleev_data = inner_shell_fingerprinter.mendeleev_data
        compositions_by_mpid = inner_shell_fingerprinter.compositions_by_mpid
        docs = inner_shell_fingerprinter.docs

        for doc in docs:
            mpid = doc['mpid']
            composition = compositions_by_mpid[mpid]
            for element in composition:
                assert mendeleev_data[element] == getattr(mendeleev, element)


    @pytest.mark.baseline
    def test_to_create_median_adsorption_energies(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_inner_shell.pkl', 'wb') as file_handle:
            pickle.dump(inner_shell_fingerprinter.median_adsorption_energies, file_handle)
        assert True


    def test__calculate_median_adsorption_energies(self, inner_shell_fingerprinter):
        with open(REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_inner_shell.pkl', 'rb') as file_handle:
            expected_median_adsorption_energies = pickle.load(file_handle)
        median_adsorption_energies = inner_shell_fingerprinter.median_adsorption_energies
        assert median_adsorption_energies == expected_median_adsorption_energies


    def test__filter_out_alloys_from_docs(self, inner_shell_fingerprinter):
        compositions_by_mpid = inner_shell_fingerprinter.compositions_by_mpid
        filtered_docs, monometallic_elements = inner_shell_fingerprinter._filter_out_alloys_from_docs()

        # Test that the function is filtering correctly
        elements = set()
        for doc in filtered_docs:
            composition = compositions_by_mpid[doc['mpid']]
            number_of_elements = len(composition)
            assert number_of_elements == 1

            # Test that we got all of the monometallic elements (and only the monometallics)
            if number_of_elements == 1:
                elements.add(composition[0])
        assert elements == monometallic_elements


    @pytest.mark.baseline
    def test_to_create_docs_fingerprints(self, inner_shell_fingerprinter):
        '''
        The arguments should really be parametrized, but pytest can't handle
        both fixtures and parametrization simultaneously. So we'll settle
        for a single test case for now.
        '''
        docs = get_catalog_docs()
        flatten = True
        fingerprints = inner_shell_fingerprinter.fingerprint_docs(docs, flatten=flatten)

        with open(REGRESSION_BASELINES_LOCATION + 'fingerprints_inner_shell.pkl', 'wb') as file_handle:
            pickle.dump(fingerprints, file_handle)
        assert True


    def test_to_fingerprint_docs(self, inner_shell_fingerprinter):
        docs = get_catalog_docs()
        flatten = True
        fingerprints = inner_shell_fingerprinter.fingerprint_docs(docs, flatten=flatten)

        with open(REGRESSION_BASELINES_LOCATION + 'fingerprints_inner_shell.pkl', 'rb') as file_handle:
            expected_fingerprints = pickle.load(file_handle)
        assert fingerprints == expected_fingerprints
