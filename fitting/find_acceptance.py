#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from Team7_Code.core import load_file, RAWFILES
from Team7_Code.ES_functions.Compiled import selection_all

from functions import calc_ctl_bins, calc_ebins_cnt, make_Leg, \
acceptance_function, rescale_q2

plt.rcParams['font.size'] = 18


# params_dict contains both user-set and calculated values,
# which are used later on in code

params_dict = {
    # q² bins for calculating the continuous fitting function
    'q2_eval': np.array(
        [0.1, 0.98, 1.1, 2.5, 4, 6, 8, 10, 12, 14, 15, 17, 19]),

    # percentiles to find ctl (inner) boundaries at
    'ctl_percentile': np.arange(10, 100, 10),

    # order of Legendre polynomial to fit to
    'i_range': 5,
    'j_range': 4,
}



# make folder called tmp
# this will be used to store intermediary variables
os.makedirs(os.path.dirname('./tmp/'), exist_ok=True)

# load required dataframes
acceptance = load_file(RAWFILES.ACCEPTANCE)
summed_dataset = pd.concat([
    *[load_file(file_name) for file_name in RAWFILES.peaking_bks],
    load_file(RAWFILES.SIGNAL)
]) # all signal + peaking in one dataframe
raw_signal = load_file(RAWFILES.SIGNAL)

filtered_summed, _ = selection_all(summed_dataset)

# save the above variables to avoid re-calculating and faster loading
with open('tmp/af_dfs.pkl', 'wb') as acceptance_params:
    pickle.dump([acceptance, filtered_summed, raw_signal], acceptance_params)



## Checkpoint 1
with open('tmp/af_dfs.pkl', 'rb') as acceptance_params:
    acceptance, filtered_summed, raw_signal = pickle.load(acceptance_params)

acceptance_func_dfs = pd.concat([acceptance, filtered_summed, raw_signal])

ctl_bins = calc_ctl_bins(
    acceptance_func_dfs, params_dict['q2_eval'], params_dict['ctl_percentile'])

params_dict['ctl_bins'] = ctl_bins

# ebins = q² bins used for evaluating the acceptance function, not to be
# confused with the bins given in the 'SM Table' on the TBPS website
acc_cnt = calc_ebins_cnt(
    acceptance, params_dict['q2_eval'], params_dict['ctl_bins'])
fil_cnt = calc_ebins_cnt(
    filtered_summed, params_dict['q2_eval'], params_dict['ctl_bins'])
sig_cnt = calc_ebins_cnt(
    raw_signal, params_dict['q2_eval'], params_dict['ctl_bins'])

# discrete acceptance function
acceptance_func_discrete = acc_cnt * fil_cnt/sig_cnt

params_dict['acceptance_func_discrete'] = acceptance_func_discrete

with open('tmp/bins_counts.pkl', 'wb') as f:
    pickle.dump(params_dict, f)



## Checkpoint 2
with open('tmp/bins_counts.pkl', 'rb') as f:
    params_dict = pickle.load(f)

# prep eq2 to use in eqn 5
# normalise q2_eval to range [-1, 1]:
eq2_normal = rescale_q2(params_dict['q2_eval'])
eq2_mid = (eq2_normal[:-1]+eq2_normal[1:])/2 # midpoint of q² bins
eq2_bw = np.diff(eq2_normal)

params_dict['eq2_mid'] = eq2_mid

# prep ctl to use in eqn 5
ctl_normal = params_dict['ctl_bins']
ctl_mid = (ctl_normal[:,:-1]+ctl_normal[:,1:])/2
ctl_bw = np.diff(ctl_normal)

params_dict['ctl_mid'] = ctl_mid

# divide discrete acceptance function by 2d bin widths
bw_2d = ctl_bw * eq2_bw[:,None]
params_dict['acceptance_func_discrete'] /= bw_2d

# list of Legendre polynomials
P = make_Leg(max(params_dict['i_range'], params_dict['j_range']))
params_dict['P'] = P

# c will store coefficients
c = np.zeros([params_dict['i_range'], params_dict['j_range']])

for i in range(params_dict['i_range']):
    for j in range(params_dict['j_range']):
        c[i, j] = (2*i+1)/2 * (2*j+1)/2 * (
            (eq2_bw*P[i](eq2_mid))[:,None] *
            ctl_bw*P[j](ctl_mid) *
            params_dict['acceptance_func_discrete']
        ).sum()


params_dict['c'] = c

with open('tmp/acceptance_coeff.pkl', 'wb') as f:
    pickle.dump(params_dict, f)


if __name__=='__main__':
    ## Checkpoint 3
    with open('tmp/acceptance_coeff.pkl', 'rb') as f:
        params_dict = pickle.load(f)

    # used to plot continuous acceptance function
    q2_range, ctl_range = np.ogrid[-1:1:100j, -1:1:100j]

    a = acceptance_function(q2_range, ctl_range, params_dict)

    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'}, constrained_layout=True)

    # loop and plot actual values
    for row in range(params_dict['ctl_mid'].shape[0]):
        ax.scatter(
            params_dict['eq2_mid'][row], params_dict['ctl_mid'][row],
            params_dict['acceptance_func_discrete'][row]
        )

    # for comparison, our function
    ax.plot_wireframe(q2_range, ctl_range, a, alpha=0.2)

    ax.set(
        xlabel='q²',
        ylabel=r'cos($\theta_l$)',
        zlabel='Count',
        # title='Count in normalised 2d bins'
    )
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.dist = 11.5
    plt.show()

