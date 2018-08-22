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
from ..fingerprints import Fingerprinter, \
    InnerShellFingerprinter, \
    OuterShellFingerprinter

# Things we need to do the tests
import os
import pytest
import numpy.testing as npt
import pickle
import json
import mendeleev
from pymatgen.ext.matproj import MPRester
from gaspy.gasdb import get_catalog_docs
from gaspy.utils import read_rc

REGRESSION_BASELINES_LOCATION = '/home/GASpy/GASpy_regressions/gaspy_regress/tests/regression_baselines/fingerprints/'


@pytest.fixture(params=['CO', 'H'], scope='module')
def fingerprinting_fixture(request):
    '''
    Note that we set this fixture's scope to 'module' so that we only
    have to make one fingerprinter, not because the tests necessarily
    need to interact with each other.
    '''
    try:
        # Remove the cache to make sure that the new instance of the fingerprinter
        # is the one that's actually making the cache correctly
        os.remove('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl')
    except OSError:
        pass

    adsorbate = request.param
    return Fingerprinter(adsorbate=adsorbate), adsorbate


class TestFingerprinter(object):
    def test___init__(self, fingerprinting_fixture):
        '''
        We only test for the presence of the attributes. The other tests
        in this class verify the correctness of these attributes.
        '''
        fingerprinter, adsorbate = fingerprinting_fixture
        assert fingerprinter.adsorbate == adsorbate
        assert hasattr(fingerprinter, 'dummy_fp')
        assert hasattr(fingerprinter, 'max_num_species')
        assert not hasattr(fingerprinter, 'adsorption_docs')
        assert not hasattr(fingerprinter, 'catalog_docs')


    @pytest.mark.baseline
    def test_to_create_dummy_fp(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        dummy_fp = fingerprinter.dummy_fp

        cache_location = REGRESSION_BASELINES_LOCATION + 'dummy_fp_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(dummy_fp, file_handle)


    def test__calculate_dummy_fp(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        dummy_fp = fingerprinter.dummy_fp

        cache_location = REGRESSION_BASELINES_LOCATION + 'dummy_fp_%s.pkl' % adsorbate
        with open(cache_location, 'rb') as file_handle:
            expected_dummy_fp = pickle.load(file_handle)
        assert dummy_fp == expected_dummy_fp


    def test__get_compositions_by_mpid(self, fingerprinting_fixture):
        '''
        3-part test:
            Test that the method saved the cache
            Test that all the MPIDs we need are there
            Test that all the data in the cache is correct
        '''
        fingerprinter, _ = fingerprinting_fixture

        # Make sure that we saved the object correctly
        with open('/home/GASpy/GASpy_regressions/cache/mp_comp_data.pkl', 'rb') as file_handle:
            saved_compositions_by_mpid = pickle.load(file_handle)
        compositions_by_mpid = fingerprinter.compositions_by_mpid
        assert compositions_by_mpid == saved_compositions_by_mpid

        # Make sure that all of the required MPIDs are in the new object
        required_mpids = set(doc['mpid'] for doc in get_catalog_docs())
        known_mpids = set(fingerprinter.compositions_by_mpid.keys())
        assert required_mpids.issubset(known_mpids)

        # Make sure that the compositions we do have are correct
        with MPRester(read_rc('matproj_api_key')) as rester:
            for mpid in known_mpids:
                entry = rester.get_entry_by_material_id({'task_ids': mpid})
                expected_composition = list(entry.as_dict()['composition'].keys())
                assert compositions_by_mpid[mpid] == expected_composition


    def test__get_mendeleev_data(self, fingerprinting_fixture):
        fingerprinter, _ = fingerprinting_fixture
        mendeleev_data = fingerprinter.mendeleev_data

        # Identify the elements that should be in `mendeleev_data`
        mpids = set(doc['mpid'] for doc in get_catalog_docs())
        elements = []
        for mpid in mpids:
            composition = fingerprinter.compositions_by_mpid[mpid]
            elements.extend(composition)
        elements = set(elements)

        # Test that we got the data correctly
        for element in elements:
            assert mendeleev_data[element] == getattr(mendeleev, element)


    @pytest.mark.baseline
    def test_to_create_median_adsorption_energies(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        median_adsorption_energies = fingerprinter.median_adsorption_energies

        cache_location = REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_%s.json' % adsorbate
        with open(cache_location, 'w') as file_handle:
            json.dump(median_adsorption_energies, file_handle)
        assert True


    def test__calculate_median_adsorption_energies(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        median_adsorption_energies = fingerprinter.median_adsorption_energies

        cache_location = REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_%s.json' % adsorbate
        with open(cache_location, 'r') as file_handle:
            expected_median_adsorption_energies = json.load(file_handle)
        assert median_adsorption_energies == expected_median_adsorption_energies


    @pytest.mark.baseline
    def test_to_create_max_num_species(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        max_num_species = fingerprinter.max_num_species

        cache_location = REGRESSION_BASELINES_LOCATION + 'max_species_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(max_num_species, file_handle)


    def test__get_max_num_species(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        max_num_species = fingerprinter.max_num_species

        cache_location = REGRESSION_BASELINES_LOCATION + 'max_species_%s.pkl' % adsorbate
        with open(cache_location, 'rb') as file_handle:
            expected_max_num_species = pickle.load(file_handle)
        assert max_num_species == expected_max_num_species


@pytest.fixture(params=['CO', 'H'], scope='module')
def inner_shell_fingerprinting_fixture(request):
    '''
    Note that we set this fixture's scope to 'module' so that we only
    have to make one fingerprinter, not because the tests necessarily
    need to interact with each other.
    '''
    adsorbate = request.param
    return InnerShellFingerprinter(adsorbate=adsorbate), adsorbate


class TestInnerShellFingerprinter(object):
    @pytest.mark.baseline
    def test_to_create_docs_fingerprints(self, inner_shell_fingerprinting_fixture):
        '''
        The arguments should really be parametrized, but pytest can't handle
        both fixtures and parametrization simultaneously. So we'll settle
        for a single test case for now.
        '''
        fingerprinter, adsorbate = inner_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.fingerprint_docs(docs)

        cache_location = REGRESSION_BASELINES_LOCATION + 'inner_shell_fingerprint_docs_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(fingerprints, file_handle)
        assert True


    def test_to_fingerprint_docs(self, inner_shell_fingerprinting_fixture):
        fingerprinter, adsorbate = inner_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.fingerprint_docs(docs)

        cache_location = REGRESSION_BASELINES_LOCATION + 'inner_shell_fingerprint_docs_%s.pkl' % adsorbate
        with open(cache_location, 'rb') as file_handle:
            expected_fingerprints = pickle.load(file_handle)
        for fingerprint, expected_fingerprint in zip(fingerprints, expected_fingerprints):
            npt.assert_allclose(fingerprint, expected_fingerprint)


@pytest.fixture(params=['CO', 'H'], scope='module')
def outer_shell_fingerprinting_fixture(request):
    '''
    Note that we set this fixture's scope to 'module' so that we only
    have to make one fingerprinter, not because the tests necessarily
    need to interact with each other.
    '''
    adsorbate = request.param
    return OuterShellFingerprinter(adsorbate=adsorbate), adsorbate


class TestOuterShellFingerprinter(object):
    @pytest.mark.baseline
    def test_to_create_docs_fingerprints(self, outer_shell_fingerprinting_fixture):
        '''
        The arguments should really be parametrized, but pytest can't handle
        both fixtures and parametrization simultaneously. So we'll settle
        for a single test case for now.
        '''
        fingerprinter, adsorbate = outer_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.fingerprint_docs(docs)

        cache_location = REGRESSION_BASELINES_LOCATION + 'outer_shell_fingerprint_docs_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(fingerprints, file_handle)
        assert True


    def test_to_fingerprint_docs(self, outer_shell_fingerprinting_fixture):
        fingerprinter, adsorbate = outer_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.fingerprint_docs(docs)

        cache_location = REGRESSION_BASELINES_LOCATION + 'outer_shell_fingerprint_docs_%s.pkl' % adsorbate
        with open(cache_location, 'rb') as file_handle:
            expected_fingerprints = pickle.load(file_handle)
        for fingerprint, expected_fingerprint in zip(fingerprints, expected_fingerprints):
            npt.assert_allclose(fingerprint, expected_fingerprint)
