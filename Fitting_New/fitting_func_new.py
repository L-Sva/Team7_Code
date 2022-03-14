#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit

from core import RAWFILES, load_file
from ES_functions.modifiedselectioncuts import selection_all
from ml_selector import remove_combinatorial_background
from Fitting_New.functions_new import (acceptance_function, log_likelihood,
log_likelihood_S, q2_binned)

plt.rcParams['font.size'] = 18
plt.rcParams['figure.constrained_layout.use'] = True


def test_NLL():
    ## Test log likelihood here for bin 0:

    _test_bin = 2
    _test_afb = 0.7
    _test_fl = 0.0

    x = np.linspace(-1, 1, 50)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # testing log_likelihood for different values of fl from -1 to 1
    ax1.plot(x, [df_log_likelihood(fl=i, afb=_test_afb, _bin=_test_bin)
    for i in x])
    ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
    ax1.set_xlabel(r'$F_L$')
    ax1.set_ylabel(r'$-\mathcal{L}$')
    ax1.grid()

    # testing log_likelihood for different values of afb from -1 to 1
    ax2.plot(x, [df_log_likelihood(fl=_test_fl, afb=i, _bin=_test_bin)
    for i in x])
    ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
    ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
    ax2.set_xlabel(r'$A_{FB}$')
    ax2.set_ylabel(r'$-\mathcal{L}$')
    ax2.grid()
    plt.show()

def fit_bins():
    ## Fitting for each bin

    # set to a negative log likelihood function
    df_log_likelihood.errordef = Minuit.LIKELIHOOD
    decimal_places = 3 # only used for outputting results
    starting_point = [-0.1, 0.0] # starting for fl and afb respectively

    # setting up lists for outputting of results
    fls, fl_errs = [], []
    afbs, afb_errs = [], []

    for i in range(10):
        m = Minuit(df_log_likelihood, fl=starting_point[0],
        afb=starting_point[1], _bin=i)

        m.fixed['_bin'] = True # don't optimise _bin

        m.limits=((-1.0, 1.0), (-1.0, 1.0), None) # set limits
        m.migrad() # find min using gradient descent
        m.hesse() # find estimation of errors at min point using Hessian

        fls.append(m.values[0])
        afbs.append(m.values[1])
        fl_errs.append(m.errors[0])
        afb_errs.append(m.errors[1])

        colour = '\033[32m' if m.fmin.is_valid else '\033[31m'

        print(
            f'Bin {i}: {np.round(fls[i], decimal_places)} ± '
            f'{np.round(fl_errs[i], decimal_places)},'
            f'{np.round(afbs[i], decimal_places)} ± '
            f'{np.round(afb_errs[i], decimal_places)}. '
            f'Function minimum considered valid: '
            f'{colour}{m.fmin.is_valid}\033[0m'
        )

    np.savez('../tmp/no_norm_af_fitting.npz', fls=fls, fl_errs=fl_errs,
    afbs=afbs, afb_errs=afb_errs)

def plot_against_SM():
    fits = np.load('../tmp/no_norm_af_fitting.npz')
    SM_data = np.load('../tmp/SM_data.npz')

    formatting = dict(ms=5, capsize=3)

    fig, ax = plt.subplots(1, 2)

    ax[0].errorbar(
        range(10), fits['fls'], yerr=fits['fl_errs'],
        fmt='k.', **formatting
    )
    ax[0].errorbar(
        range(10), SM_data['FL'], yerr=SM_data['FL_err'],
        fmt='r.', **formatting
    )
    ax[0].set(xlabel='Bin number', ylabel=r'$F_L$', xticks=range(10))
    ax[0].grid()

    ax[1].errorbar(
        range(10), fits['afbs'], yerr=fits['afb_errs'],
        fmt='k.', **formatting
    )
    ax[1].errorbar(
        range(10), SM_data['AFB'], yerr=SM_data['AFB_err'],
        fmt='r.', **formatting
    )
    ax[1].set(xlabel='Bin number', ylabel=r'$A_{FB}$', xticks=range(10))
    ax[1].grid()

    fig.suptitle('Our fit (black), SM Prediction (red)', size=17)

    plt.show()


if __name__ == '__main__':
    '''
    # fist run to generate the files
    dataframe = load_file(RAWFILES.TOTAL_DATASET)
    dataframe, _ = selection_all(dataframe)
    dataframe, _ = remove_combinatorial_background(dataframe)
    dataframe.to_pickle('../tmp/filtered_total_dataset.pkl')
    '''

    # read file to avoid recalculation
    dataframe = pd.read_pickle('../tmp/filtered_total_dataset.pkl')
    bins = q2_binned(dataframe)
    coeff = np.load('../tmp/coeff.npy')

    df_log_likelihood = partial(log_likelihood, bins, coeff)

    fl_range = np.linspace(-1, 1, 30).reshape(-1, 1, 1)
    afb_range = np.linspace(-1, 1, 30).reshape(1, -1, 1)
    NLL_ranged = df_log_likelihood(fl_range, afb_range, 0)
    
    SM_data = np.load('../tmp/SM_data.npz')
    
    fig, ax = plt.subplots()
    axc = ax.contourf(
        fl_range.flatten(), afb_range.flatten(), NLL_ranged, cmap='binary')
    fig.colorbar(axc)
    ax.plot(SM_data['FL'][0], SM_data['AFB'][0], 'r.')
    ax.set(xlabel=r'$F_L$', ylabel=r'$A_{FB}$')
    ax.set_title('Not normalised, bin 0')
    plt.show()


    # test_NLL()
    # fit_bins()
    # plot_against_SM()


    # 8-param fits:
    # bins_log_likelihood_S = partial(log_likelihood_S, bins, coeff)

    # bins_log_likelihood_S.errordef = Minuit.LIKELIHOOD

    # bin_no = 3 # change value here
    # results = []
    # errors = []

    # starting_point = [
    #     0.7112903962261427, 0.12215506287920677, -0.024751439845413715,
    #     -0.22420373935504972, -0.3371396761500003, -0.013382849845922798,
    #     -0.0050616569813558675, -0.00070641680143312
    # ]
    # m = Minuit(bins_log_likelihood_S, *starting_point, bin_no)

    # m.fixed['_bin'] = True # don't optimise _bin
    # m.limits=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
    #           (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), None)
    # m.migrad()
    # results.append(np.array(m.values))
    # errors.append(np.array(m.errors))
    # # m.fmin
    # # m.params

    # msg = '\033[32mValid' if m.fmin.is_valid else '\033[31mNot Valid'
    # print(f'Bin {bin_no}: {msg}\033[0m')

    # print(results, errors)

