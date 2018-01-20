'''
This module contains our current "default" methods for doing various things,
such as modeling or predicting and then saving them.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from sklearn.gaussian_process import GaussianProcessRegressor
from tpot import TPOTRegressor
import gaspy_regress.gio
import gaspy_regress.predict
from gaspy_regress.regressor import GASpyRegressor
from gaspy.utils import vasp_settings_to_str, read_rc


def modeling():
    VASP_SETTINGS = vasp_settings_to_str({'gga': 'RP',
                                          'pp_version': '5.4',
                                          'encut': 350})
    model_name = 'GP_around_TPOT'
    features = ['coordcount']
    outer_features = ['neighbors_coordcounts']
    responses = ['energy']
    blocks = ['adsorbate']
    fingerprints = {'neighborcoord': '$processed_data.fp_final.neighborcoord'}
    tpot = TPOTRegressor(generations=8,
                         population_size=32,
                         verbosity=1,
                         random_state=42)
    gp = GaussianProcessRegressor()
    H = GASpyRegressor(features=features, responses=responses,
                       blocks=blocks, vasp_settings=VASP_SETTINGS,
                       fingerprints=fingerprints, train_size=1)
    H.fit_tpot(tpot, model_name=model_name)
    H.fit_hierarchical(gp, 'fit_sk', outer_features, model_name=model_name)
    gaspy_regress.gio.dump_model(H)


def prediction():
    model_name = 'GP_around_TPOT'
    features = ['coordcount']
    outer_features = ['neighbors_coordcounts']
    responses = ['energy']
    blocks = ['adsorbate']
    H = gaspy_regress.gio.load_model(model_name, features+outer_features, responses, blocks)
    excel_file_path = read_rc()['gaspy_path'] + '/GASpy_regressions/volcanos_parsed.xlsx'
    regressor_block = ('CO',)
    adsorbate = 'CO'
    system = 'CO2RR'
    scale = 'log'
    co2_data = gaspy_regress.predict.volcano(H, regressor_block, system, excel_file_path, scale, adsorbate)
    gaspy_regress.gio.dump_predictions(co2_data, regressor=H, system=system)
    regressor_block = ('H',)
    adsorbate = 'H'
    system = 'HER'
    scale = 'log'
    her_data = gaspy_regress.predict.volcano(H, regressor_block, system, excel_file_path, scale, adsorbate)
    gaspy_regress.gio.dump_predictions(her_data, regressor=H, system=system)
