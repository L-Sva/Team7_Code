""" Implementation of the ML background removing models as selectors
"""

import xgboost
import os
from histrogram_plots import generic_selector_plot

from ml_tools import ml_get_model_sig_prob, ml_strip_columns
from core import RAWFILES, load_file, combine_n_selectors

COMB_BK_MODEL_FILE_NAME = 'comb_hyperparameters_opt_best.model'
COMB_BK_THRESH = 0.48

PK_BK_MODEL_FILE_NAME = 'pk_hyperparameters_opt_best.model'
PK_BK_THRESH = 0.5

comb_bk_model = xgboost.XGBClassifier()
comb_bk_model.load_model(os.path.join('examples_save',COMB_BK_MODEL_FILE_NAME))

pk_bk_model = xgboost.XGBClassifier()
pk_bk_model.load_model(os.path.join('examples_save',PK_BK_MODEL_FILE_NAME))

def remove_combinatorial_background(data):
    data_sc = ml_strip_columns(data,reject_column_names=('B0_MM','Kstar_MM'))
    sig_prob = ml_get_model_sig_prob(data_sc,comb_bk_model)
    accept = sig_prob > COMB_BK_THRESH
    s = data[accept]
    ns = data[~accept]
    return s, ns

def remove_peaking_background(data):
    data_sc = ml_strip_columns(data)
    sig_prob = ml_get_model_sig_prob(data_sc, pk_bk_model)
    accept = sig_prob > PK_BK_THRESH
    s = data[accept]
    ns = data[~accept]
    return s, ns

remove_all_bk = combine_n_selectors(remove_combinatorial_background, remove_peaking_background)

if __name__ == "__main__":
    total = load_file(RAWFILES.TOTAL_DATASET)
    s, ns = remove_all_bk(total)

    generic_selector_plot(total, s, ns, 'B0_MM')