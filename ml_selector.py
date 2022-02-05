""" Implementation of the ML background removing models as selectors
"""

from math import comb
import xgboost
import os
import matplotlib.pyplot as plt
import ES_functions
from histrogram_plots import generic_selector_plot, plot_hist_quantity
from ml_tools import ml_get_model_sig_prob, ml_strip_columns
from core import RAWFILES, ensure_dir, load_file, combine_n_selectors
from ES_functions.Compiled import q2_resonances

IMAGE_OUTPUT_DIR = 'data_ml_selectors_histograms'

COMB_BK_MODEL_FILE_NAME = 'comb_hyperparameters_opt_best.model'
COMB_BK_THRESH = 0.48

PK_BK_MODEL_FILE_NAME = 'pk_hyperparameters_opt_best.model'
PK_BK_THRESH = 0.9984974958263773

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
    remove = combine_n_selectors(remove_combinatorial_background, remove_peaking_background, q2_resonances)

    total = load_file(RAWFILES.TOTAL_DATASET)
    signal = load_file(RAWFILES.SIGNAL)
    #s, ns = remove_all_bk(total)
    s, ns = remove_peaking_background(total)

    ensure_dir(IMAGE_OUTPUT_DIR)

    for column in total:
        if total[column].dtype != 'object':
            bins, h = plot_hist_quantity(total, column, label='Total dataset', bins=150)
            plot_hist_quantity(s, column, label='ML - Signal', bins = bins)
            plot_hist_quantity(ns, column, label='ML - Background', bins=bins)
            plot_hist_quantity(signal, column, label='Simulated signal', bins=bins)
            plt.legend()

            plt.savefig(
                os.path.join(IMAGE_OUTPUT_DIR,f'{column}.png')
            )
            plt.close()