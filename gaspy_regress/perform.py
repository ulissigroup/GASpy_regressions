'''
This module contains our current "default" methods for doing various things,
such as modeling or predicting and then saving them.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from tpot import TPOTRegressor
import gaspy_regress.gio
import gaspy_regress.predict
from gaspy_regress.regressor import GASpyRegressor
from gaspy.utils import vasp_settings_to_str, read_rc


def modeling(fit_blocks=None, train_size=1, dev_size=None, tpot_verbosity=1,
             tpot_generations=1, tpot_population=16, tpot_offspring=16,
             n_jobs=1, random_state=None):
    VASP_SETTINGS = vasp_settings_to_str({'gga': 'RP',
                                          'pp_version': '5.4',
                                          'encut': 350})
    model_name = 'TPOT'
    features = ['coordatoms_chemfp0', 'neighbors_chemfp0']
    responses = ['energy']
    blocks = ['adsorbate']
    tpot = TPOTRegressor(generations=tpot_generations,
                         population_size=tpot_population,
                         offspring_size=tpot_offspring,
                         verbosity=tpot_verbosity,
                         random_state=random_state,
                         n_jobs=n_jobs)
    H = GASpyRegressor(features=features, responses=responses,
                       blocks=blocks, vasp_settings=VASP_SETTINGS,
                       train_size=train_size, dev_size=dev_size,
                       dim_red='pca')
    H.fit_tpot(tpot, model_name=model_name, blocks=fit_blocks)
    gaspy_regress.gio.dump_model(H)


def prediction(CO2RR=True, HER=True, save_all_estimates=True, doc_chunk_size=1000):
    model_name = 'TPOT'
    features = ['coordatoms_chemfp0', 'neighbors_chemfp0']
    responses = ['energy']
    blocks = ['adsorbate']
    H = gaspy_regress.gio.load_model(model_name, features, responses, blocks)
    excel_file_path = read_rc()['gaspy_path'] + '/GASpy_regressions/volcanos_parsed.xlsx'
    if CO2RR:
        regressor_block = ('CO',)
        adsorbate = 'CO'
        system = 'CO2RR'
        scale = 'log'
        co2_data = gaspy_regress.predict.volcano(H, regressor_block, system,
                                                 excel_file_path, scale, adsorbate,
                                                 doc_chunk_size=doc_chunk_size,
                                                 save_all_estimates=True)
        gaspy_regress.gio.dump_predictions(co2_data, regressor=H, system=system)
    if HER:
        regressor_block = ('H',)
        adsorbate = 'H'
        system = 'HER'
        scale = 'log'
        her_data = gaspy_regress.predict.volcano(H, regressor_block, system,
                                                 excel_file_path, scale, adsorbate,
                                                 doc_chunk_size=doc_chunk_size,
                                                 save_all_estimates=True)
        gaspy_regress.gio.dump_predictions(her_data, regressor=H, system=system)
