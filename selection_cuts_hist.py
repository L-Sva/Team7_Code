from core import load_file, RAWFILES
import pandas as pd
from ES_functions.Compiled import q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all
import numpy as np
from test_candidates_example import test_candidate_true_false_positive_negative
from histrogram_plots import generic_selector_plot
import matplotlib.pyplot as plt
#%%
def ml_combine_signal_bk(signal_dataset, background_dataset):
    """Combines signal and background dataset, adding category labels
    """
    # Marek
    signal_dataset = signal_dataset.copy()
    background_dataset = background_dataset.copy()
    signal_dataset.loc[:,'category'] = 1
    background_dataset.loc[:,'category'] = 0

    # combine
    dataset = pd.concat((signal_dataset, background_dataset))
    return dataset
#%%
peaking_bk=[]
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    peaking_bk.append(data)

peaking_bk_combined=pd.concat(peaking_bk)
peaking_bk.append(peaking_bk_combined)

mod_peaking_bk=[]

signal = load_file(RAWFILES.SIGNAL)
for non_signal in peaking_bk:
    test_data=ml_combine_signal_bk(signal, non_signal)
    mod_peaking_bk.append(test_data)

funclist=[q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all]

#%%
column = 'q2'

for selection_method in funclist:
    for test_data in range(len(mod_peaking_bk)):
        s, ns = selection_method(mod_peaking_bk[test_data])
        val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = selection_method)
        print(val)
        plt.figure(test_data)
        generic_selector_plot(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = True)
        #plt.savefig()

