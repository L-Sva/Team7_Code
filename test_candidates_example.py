from core import load_file, RAWFILES
from histrogram_plots import generic_selector_plot, plot_hist_quantity
from ES_functions.ES2 import ES2
from ES_functions.ES1 import ES1, alt_ES1
from ml_tools import test_false_true_negative_positive, test_sb
import ml_load
import ml_tools
import pandas as pd
import matplotlib.pyplot as plt
#%%
def test_candidate_true_false_positive_negative(signal, non_signal, selection_method=ES1):
    dataframe = ml_tools.ml_combine_signal_bk(signal, non_signal)
    s, ns = selection_method(dataframe)
    s['pred'] = 1
    ns['pred'] = 0
    test_res_data = pd.concat((s,ns))
    sig_prob = test_res_data['pred'].to_numpy()
    
    print(test_false_true_negative_positive(test_res_data, sig_prob ,0.5))
    print(test_sb(test_res_data, sig_prob ,0.5))
    
#%%
signal = load_file(RAWFILES.SIGNAL)

non_signal = []
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    non_signal.append(data)

non_signal = pd.concat(non_signal)

test_candidate_true_false_positive_negative(signal, non_signal)