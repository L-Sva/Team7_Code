""" Implementation of the ML background removing models as selectors
"""

import os
import matplotlib.pyplot as plt
from core import RAWFILES, combine_n_selectors, load_file
from ES_functions.Compiled import q2_resonances
from ml_main import make_selector, load_model_file

IMAGE_OUTPUT_DIR = 'data_ml_selectors_histograms'
ML_SAVE_DIR = 'ml_models'

COMB_BK_MODEL_FILE_NAME = 'comb_hyperparameters_opt_best.model'
COMB_BK_PATH = os.path.join(ML_SAVE_DIR,COMB_BK_MODEL_FILE_NAME)
COMB_THRESH = 0.66

PK_BK_MODEL_FILE_NAME = 'pk_hyperparameters_opt_best.model'
PK_BK_PATH = os.path.join(ML_SAVE_DIR,PK_BK_MODEL_FILE_NAME)
PK_THRESH = 0.81

comb_bk_model = load_model_file(COMB_BK_PATH)
remove_combinatorial_background = make_selector(comb_bk_model, COMB_THRESH, 
    reject_column_names=('B0_ID','polarity','B0_MM','Kstar_MM'))

pk_bk_model = load_model_file(PK_BK_PATH)
remove_peaking_background = make_selector(pk_bk_model, PK_THRESH)
remove_all_bk = combine_n_selectors(remove_combinatorial_background, remove_peaking_background, q2_resonances)

if __name__ == '__main__':

    data = load_file(RAWFILES.TOTAL_DATASET)
    subset, notsubset = remove_all_bk(data)
    print(f'Accepted {len(subset)} events ({len(subset) / len(data) * 100:.4g}%)')