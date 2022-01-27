from core import load_file, RAWFILES
from histrogram_plots import generic_selector_plot, plot_hist_quantity
from ES_functions.ES2 import ES2
from ES_functions.ES1 import ES1, alt_ES1
from ml_tools import test_false_true_negative_positive, test_sb
import ml_load
import pandas as pd
import matplotlib.pyplot as plt

train_data, test_data = ml_load.get_train_test_for_all_peaking_bks()

s, ns = ES1(test_data)
s['pred'] = 1
ns['pred'] = 0
test_res_data = pd.concat((s,ns))
sig_prob = test_res_data['pred'].to_numpy()

print(test_false_true_negative_positive(test_res_data, sig_prob ,0.5))
print(test_sb(test_res_data, sig_prob ,0.5))

