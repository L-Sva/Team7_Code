#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:29:35 2022

@author: emirsezik
"""

import numpy as np
import pandas as pd
from scipy.integrate import dblquad
from pathlib import Path
from iminuit import Minuit
from modifiedselectioncuts import selection_all
from ml_selector import remove_combinatorial_background
from find_acceptance_new_reduced import acceptance_function

def binning(dataframe):
    q_start = [0.1, 1.1, 2.5, 4, 6, 15, 17, 11, 1, 15]
    q_end = [0.98, 2.5, 4, 6, 8, 17, 19, 12.5, 6, 17.9]
    bins = []
    q2 = dataframe["q2"]
    for i in range(len(q_start)):
        cond = (q2 >= q_start[i]) & (q2 <= q_end[i])
        bins.append(dataframe[cond])
    return bins
bin_dic = {
  0: (0.1, 0.98),
  1: (1.1, 2.5),
  2: (2.5, 4),
  3: (4, 6),
  4: (6, 8),
  5: (15, 17),
  6: (17, 19),
  7: (11, 12.5),
  8: (1, 6),
  9: (15, 17.9)
}

coeff = np.load('../tmp_redu/coeff.npy')

# =============================================================================
# #%% fist run to generate the files
# dataframe = pd.read_pickle("Data/total_dataset.pkl")
# dataframe,_  = selection_all(dataframe)
# dataframe,_ = remove_combinatorial_background(dataframe)
# dataframe.to_pickle('../tmp_redu/filtered_total_dataset.pkl')
# 
# =============================================================================
#%% read file to avoid recalculation
dataframe = pd.read_pickle('../tmp_redu/filtered_total_dataset.pkl')

#%%
bins = binning(dataframe)

#%%
def decay_rate(F_l, A_fb, acceptance, q2, ctl):
    c2tl = 2 * ctl * ctl - 1
    scalar_array = 3/8 * (3/2 - 1/2 * F_l + 1/2 * c2tl * (1 - 3 * F_l) + 8/3 * A_fb * ctl) * acceptance(q2, ctl, coeff)
    return scalar_array

def log_likelihood(F_l, A_fb, _bin):
    bin_num = int(_bin)
    _bin = bins[bin_num]
    q2 = _bin['q2']
    ctl = _bin['costhetal']
    scalar_array = decay_rate(F_l, A_fb, acceptance_function, q2, ctl)
    scalar_array = [float(i) for i in scalar_array]
    def func(q2, ctl):
        return decay_rate(F_l, A_fb, acceptance_function, q2, ctl)
    norm = dblquad(func, bin_dic[bin_num][0], bin_dic[bin_num][1], -1, 1)[0]
    return - np.sum(np.log(scalar_array)) + np.log(norm)

#%%
log_likelihood.errordef = Minuit.LIKELIHOOD

results = []
errors = []

#%%
starting_point = [0.711290, 0.122155]
m = Minuit(log_likelihood, starting_point[0], starting_point[1], 3)

m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
m.migrad()
results.append(np.array(m.values))
errors.append(np.array(m.errors))
#m.fmin
#m.params