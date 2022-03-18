# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:39:25 2022

@author: coxo
"""

from core import load_file, RAWFILES
import pandas as pd
from ES_functions.modifiedselectioncuts import q2_resonances,q2_range,K0_vertex_chi2, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD,KSTAR_FD, DIRA, Particle_ID,B0_mass, selection_all,peaking_back_1,peaking_back_2,peaking_back_3,peaking_back_4,peaking_back_5,peaking_back_6,peaking_back_7,peaking_back_8
import numpy as np
from test_candidates_example import test_candidate_true_false_positive_negative
from histrogram_plots_1 import generic_selector_plot_new
import matplotlib.pyplot as plt
#%%
#Function combines signal and peaking background events
def ml_combine_total_bk(total_dataset, background_dataset):
    """Combines signal and background dataset, adding category labels
    """
    # Marek
    total_dataset = total_dataset.copy()
    background_dataset = background_dataset.copy()
    total_dataset.loc[:,'category'] = 1
    background_dataset.loc[:,'category'] = 0

    # combine
    dataset = pd.concat((total_dataset, background_dataset))
    return dataset
#%%
#Load peaking backgrounds
peaking_bk=[]
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    peaking_bk.append(data)
#create a "combined peaking background" background by concatenating invidual ones
peaking_bk_combined=pd.concat(peaking_bk)
#peaking_bk.append(peaking_bk_combined)

# =============================================================================
# mod_peaking_bk=[]
# #combine the backgrounds with the signal
# signal = load_file(RAWFILES.SIGNAL)
# for non_signal in peaking_bk:
#     test_data=ml_combine_signal_bk(signal, non_signal)
#     mod_peaking_bk.append(test_data)
# =============================================================================
total=load_file(RAWFILES.TOTAL_DATASET)
combined=ml_combine_total_bk(total,peaking_bk_combined)