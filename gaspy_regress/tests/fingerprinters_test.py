'''
Tests for the `fingerprints` submodule.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/jovyan/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..fingerprinters import (Fingerprinter,
                              InnerShellFingerprinter,
                              OuterShellFingerprinter,
                              StackedFingerprinter)

# Things we need to do the tests
import os
import pytest
import numpy.testing as npt
import copy
import pickle
import json
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='numpy.dtype size changed')
    import mendeleev
from pymatgen.ext.matproj import MPRester
from gaspy.gasdb import get_adsorption_docs, get_catalog_docs
from gaspy.utils import read_rc

REGRESSION_BASELINES_LOCATION = ('/home/jovyan/GASpy/GASpy_regressions/'
                                 'gaspy_regress/tests/regression_baselines/fingerprinters/')
MP_CACHE_LOCATION = read_rc('gasdb_path') + '/mp_comp_data.pkl'

# Ignore some warnings that we know will come up
pytestmark = pytest.mark.filterwarnings('ignore: We do not have any energy data for',
                                        'ignore: Mean of empty slice',
                                        'ignore: invalid value encountered in double_scalars',
                                        'ignore: You are using adsorption document filters for a '
                                        'set of adsorbates that we have not yet established valid '
                                        'energy bounds for, yet. We are accepting anything in the '
                                        'range between')


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
        os.remove(MP_CACHE_LOCATION)
    except OSError:
        pass

    # Initialize and train a fingerprinter
    adsorbate = request.param
    docs = get_adsorption_docs([adsorbate])
    fingerprinter = Fingerprinter().fit(docs)
    return fingerprinter, adsorbate


class TestFingerprinter(object):
    def test_fit(self, fingerprinting_fixture):
        '''
        We only test for the presence of the attributes. The other tests
        in this class verify the correctness of these attributes.
        '''
        fingerprinter, _ = fingerprinting_fixture
        assert hasattr(fingerprinter, 'dummy_fp_')
        assert hasattr(fingerprinter, 'max_num_species_')
        assert not hasattr(fingerprinter, 'adsorption_docs')
        assert not hasattr(fingerprinter, 'catalog_docs')


    @pytest.mark.baseline
    def test_to_create_dummy_fp(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        dummy_fp = fingerprinter.dummy_fp_

        cache_location = REGRESSION_BASELINES_LOCATION + 'dummy_fp_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(dummy_fp, file_handle)


    def test__calculate_dummy_fp(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        dummy_fp = fingerprinter.dummy_fp_

        cache_location = REGRESSION_BASELINES_LOCATION + 'dummy_fp_%s.pkl' % adsorbate
        with open(cache_location, 'rb') as file_handle:
            expected_dummy_fp = pickle.load(file_handle)
        for value, expected_value in zip(dummy_fp, expected_dummy_fp):
            assert round(value, 7) == round(expected_value, 7)


    def test__get_compositions_by_mpid(self, fingerprinting_fixture):
        '''
        3-part test:
            Test that the method saved the cache
            Test that all the MPIDs we need are there
            Test that all the data in the cache is correct
        '''
        fingerprinter, _ = fingerprinting_fixture

        # Make sure that we saved the object correctly
        with open(MP_CACHE_LOCATION, 'rb') as file_handle:
            saved_compositions_by_mpid = pickle.load(file_handle)
        compositions_by_mpid = fingerprinter.compositions_by_mpid_
        assert compositions_by_mpid == saved_compositions_by_mpid

        # Make sure that all of the required MPIDs are in the new object
        required_mpids = set(doc['mpid'] for doc in get_catalog_docs())
        known_mpids = set(fingerprinter.compositions_by_mpid_.keys())
        assert required_mpids.issubset(known_mpids)

        # Make sure that the compositions we do have are correct
        with MPRester(read_rc('matproj_api_key')) as rester:
            for mpid in known_mpids:
                entry = rester.get_entry_by_material_id({'task_ids': mpid})
                expected_composition = list(entry.as_dict()['composition'].keys())
                assert set(compositions_by_mpid[mpid]) == set(expected_composition)


    def test__get_elements_in_scope(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        elements = fingerprinter.elements_

        required_mpids = set(doc['mpid'] for doc in get_adsorption_docs(adsorbates=[adsorbate])) | \
                         set(doc['mpid'] for doc in get_catalog_docs())
        expected_elements = []
        for mpid in required_mpids:
            composition = fingerprinter.compositions_by_mpid_[mpid]
            expected_elements.extend(composition)
        expected_elements = set(expected_elements)

        assert expected_elements.issubset(elements)


    def test__get_mendeleev_data(self, fingerprinting_fixture):
        fingerprinter, _ = fingerprinting_fixture
        mendeleev_data = fingerprinter.mendeleev_data_

        # Identify the elements that should be in `mendeleev_data`
        mpids = set(doc['mpid'] for doc in get_catalog_docs())
        elements = []
        for mpid in mpids:
            composition = fingerprinter.compositions_by_mpid_[mpid]
            elements.extend(composition)
        elements = set(elements)

        # Test that we got the data correctly
        for element in elements:
            assert mendeleev_data[element] == getattr(mendeleev, element)


    @pytest.mark.baseline
    def test_to_create_median_adsorption_energies(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        median_adsorption_energies = fingerprinter.median_adsorption_energies_

        cache_location = REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_%s.json' % adsorbate
        with open(cache_location, 'w') as file_handle:
            json.dump(median_adsorption_energies, file_handle)
        assert True


    def test__calculate_median_adsorption_energies(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        median_adsorption_energies = fingerprinter.median_adsorption_energies_

        cache_location = REGRESSION_BASELINES_LOCATION + 'median_adsorption_energies_%s.json' % adsorbate
        with open(cache_location, 'r') as file_handle:
            expected_median_adsorption_energies = json.load(file_handle)
        assert median_adsorption_energies == expected_median_adsorption_energies


    @pytest.mark.baseline
    def test_to_create_max_num_species(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        max_num_species = fingerprinter.max_num_species_

        cache_location = REGRESSION_BASELINES_LOCATION + 'max_species_%s.pkl' % adsorbate
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(max_num_species, file_handle)


    def test__get_max_num_species(self, fingerprinting_fixture):
        fingerprinter, adsorbate = fingerprinting_fixture
        max_num_species = fingerprinter.max_num_species_

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
    docs = get_adsorption_docs([adsorbate])
    fingerprinter = InnerShellFingerprinter().fit(docs)
    return fingerprinter, adsorbate


class TestInnerShellFingerprinter(object):
    @pytest.mark.baseline
    def test_to_create_transformations(self, inner_shell_fingerprinting_fixture):
        '''
        The arguments should really be parametrized, but pytest can't handle
        both fixtures and parametrization simultaneously. So we'll settle
        for a single test case for now.
        '''
        fingerprinter, adsorbate = inner_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.transform(docs)

        cache_location = (REGRESSION_BASELINES_LOCATION +
                          'inner_shell_fingerprinter_transformation_%s.pkl' % adsorbate)
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(fingerprints, file_handle)
        assert True


    def test_transform(self, inner_shell_fingerprinting_fixture):
        fingerprinter, adsorbate = inner_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.transform(docs)

        cache_location = (REGRESSION_BASELINES_LOCATION +
                          'inner_shell_fingerprinter_transformation_%s.pkl' % adsorbate)
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
    docs = get_adsorption_docs([adsorbate])
    fingerprinter = OuterShellFingerprinter().fit(docs)
    return fingerprinter, adsorbate


class TestOuterShellFingerprinter(object):
    @pytest.mark.baseline
    def test_to_create_transformations(self, outer_shell_fingerprinting_fixture):
        '''
        The arguments should really be parametrized, but pytest can't handle
        both fixtures and parametrization simultaneously. So we'll settle
        for a single test case for now.
        '''
        fingerprinter, adsorbate = outer_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.transform(docs)

        cache_location = (REGRESSION_BASELINES_LOCATION +
                          'outer_shell_fingerprinter_transformation_%s.pkl' % adsorbate)
        with open(cache_location, 'wb') as file_handle:
            pickle.dump(fingerprints, file_handle)
        assert True


    def test_transform(self, outer_shell_fingerprinting_fixture):
        fingerprinter, adsorbate = outer_shell_fingerprinting_fixture
        docs = get_catalog_docs()
        fingerprints = fingerprinter.transform(docs)

        cache_location = (REGRESSION_BASELINES_LOCATION +
                          'outer_shell_fingerprinter_transformation_%s.pkl' % adsorbate)
        with open(cache_location, 'rb') as file_handle:
            expected_fingerprints = pickle.load(file_handle)
        for fingerprint, expected_fingerprint in zip(fingerprints, expected_fingerprints):
            npt.assert_allclose(fingerprint, expected_fingerprint)


    def test__concatenate_second_shell(self, outer_shell_fingerprinting_fixture):
        fingerprinter, _ = outer_shell_fingerprinting_fixture
        doc = {'neighborcoord': ['Cu:Cu-Cu-Cu-Cu-Cu-Al',
                                 'Al:Cu-Cu-Cu-Cu-Cu-Cu']}
        second_shell_atoms = fingerprinter._concatenate_second_shell(doc)
        assert second_shell_atoms == ['Cu']*5 + ['Al'] + ['Cu']*6


@pytest.mark.parametrize('fingerprinters,collection_to_fp',
                         [((InnerShellFingerprinter(), OuterShellFingerprinter()), 'adsorption'),
                          ((InnerShellFingerprinter(), OuterShellFingerprinter()), 'adsorption'),
                          ((InnerShellFingerprinter(), OuterShellFingerprinter()), 'catalog'),
                          ((InnerShellFingerprinter(), OuterShellFingerprinter()), 'catalog')])
def test_StackedFingerprinter(fingerprinters, collection_to_fp):
    '''
    Note that the argument `fingerprinters` can be a tuple of any size.
    In other words:  We hope to be able to stack any number of fingerprinters.
    '''
    # Get the training and test cases
    docs_train = get_adsorption_docs()
    if collection_to_fp == 'adsorption':
        docs_test = get_adsorption_docs()
    elif collection_to_fp == 'catalog':
        docs_test = get_catalog_docs()

    # Call the function we're testing
    stacked_fingerprinter = StackedFingerprinter(*(copy.deepcopy(fingerprinter)
                                                   for fingerprinter in fingerprinters))
    stacked_fingerprinter.fit(docs_train)
    stacked_fingerprints = stacked_fingerprinter.transform(docs_test)

    # Compare to what we expect it to be
    for fingerprinter in fingerprinters:
        fingerprinter.fit(docs_train)
    tupled_fingerprints = tuple(fingerprinter.transform(docs_test)
                                for fingerprinter in fingerprinters)
    expected_stacked_fingerprints = np.concatenate(tupled_fingerprints, axis=1)
    npt.assert_allclose(stacked_fingerprints, expected_stacked_fingerprints)
