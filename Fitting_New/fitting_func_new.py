#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:29:35 2022

@author: emirsezik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vegas
from pathlib import Path
from iminuit import Minuit
from modifiedselectioncuts import selection_all
from ml_selector import remove_combinatorial_background
from find_acceptance_new import acceptance_function

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

coeff = np.load('../tmp/coeff.npy')

#%% fist run to generate the files
dataframe = pd.read_pickle("Data/total_dataset.pkl")
dataframe,_  = selection_all(dataframe)
dataframe,_ = remove_combinatorial_background(dataframe)
dataframe.to_pickle('../tmp/filtered_total_dataset.pkl')

#%% read file to avoid recalculation
dataframe = pd.read_pickle('../tmp/filtered_total_dataset.pkl')

#%%
bins = binning(dataframe)

#%%
def decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9, acceptance, q2, ctl, ctk, phi):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    stl = np.sqrt(1 - ctl * ctl)
    stk = np.sqrt(1 - ctk * ctk)
    c2tl = 2 * ctl * ctl - 1
    s2tk = 2 * stk * ctk
    s2tl = 2 * stl * ctl
    stl_sq = stl * stl
    stk_sq = stk * stk
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    
    scalar_array = 9 * np.pi / 32 * acceptance(q2, ctl, ctk, phi, coeff) * (3/4 * (1 - F_l) * stk_sq +
                                                  F_l * ctk * ctk +
                                                  1/4 * (1 - F_l) * stk_sq * c2tl - 
                                                  F_l * ctk * ctk * c2tl + 
                                                  S_3 * stk_sq * stl_sq * np.cos(2 * phi) +
                                                  S_4 * s2tk * s2tl * cphi + 
                                                  S_5 * s2tk * stl * cphi + 
                                                  4/3 * A_fb * stk_sq * ctl +
                                                  S_7 * s2tk * stl * sphi +
                                                  S_8 * s2tk * s2tl * sphi +
                                                  S_9 * stk_sq * stl_sq * 2 * sphi * cphi
                                                  )
    return scalar_array


def log_likelihood_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    binnum = int(_bin)
    _bin = bins[binnum]
    ctl = _bin['costhetal']
    ctk = _bin["costhetak"]
    phi = _bin["phi"]
    q2 = _bin["q2"]
    normalised_scalar_array = decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9, acceptance_function, q2, ctl, ctk, phi)
    def int_func(x):
        return decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9, acceptance_function, x[0], x[1], x[2], x[3])
    norm = vegas.Integrator([[bin_dic[binnum][0], bin_dic[binnum][1]], [-1, 1], [-1, 1], [-np.pi, np.pi]])
    result = norm(int_func, nitn=10, neval=100)
    normalised_scalar_array = np.array([float(i) for i in normalised_scalar_array]) / result.mean

    return -np.sum(np.log(normalised_scalar_array))

#%%
log_likelihood_S.errordef = Minuit.LIKELIHOOD

results = []
errors = []

#%%
starting_point = [0.711290, 0.122155, -0.024751, -0.224204, -0.337140, -0.013383,-0.005062,-0.000706]
m = Minuit(log_likelihood_S, starting_point[0], starting_point[1], starting_point[2], 
               starting_point[3], starting_point[4], starting_point[5], 
               starting_point[6], starting_point[7], 1)
    
m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
m.limits=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
              (-1.0, 1.0), (-1.0, 1.0), None)
m.migrad()
results.append(np.array(m.values))
errors.append(np.array(m.errors))
#m.fmin
#m.params
