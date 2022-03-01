#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy.polynomial import Legendre

q2bins = np.array(
    [[0.1, 0.98],
     [1.1, 2.5],
     [2.5, 4.0],
     [4.0, 6.0],
     [6.0, 8.0],
     [15.0, 17.0],
     [17.0, 19.0],
     [11.0, 12.5],
     [1.0, 6.0],
     [15.0, 17.9]]
)

def q2_binned(df):
    """
    Bin a given dataframe into q² bins.

    Parameters
    ----------
    df : dataframe/numpy array
        dataset to split into q² bins

    Returns
    -------
    dictionary
        with keys from 0..9, corresponding to the q² bins given in the TBPS
        website, also contains i1,i2,i3,i4, which are 'invalid' bins
    """

    # bins ordered different, because np.histogram needs a
    # monotonically increasing array
    q2_bins_0 = [0.1, 0.98, 1.1, 2.5, 4, 6, 8, 15, 17, 19]
    q2_bins_1 = [1, 6, 11, 12.5, 15, 17.9]

    binned_0 = pd.cut(
        df['q2'], bins=q2_bins_0,
        labels=['0', 'i1', '1', '2', '3', '4', 'i2', '5', '6']
    )
    binned_1 = pd.cut(
        df['q2'], bins=q2_bins_1,
        labels=['8', 'i3', '7', 'i4', '9']
    )

    df_q2_bins_0 = dict(tuple(df.groupby(binned_0)))
    df_q2_bins_1 = dict(tuple(df.groupby(binned_1)))

    df_q2_binned = {**df_q2_bins_0, **df_q2_bins_1}

    return df_q2_binned

def calc_ctl_bins(acceptance_func_dfs, q2_eval, ctl_percentile):
    # used to choose ctl bins by percentiles
    eval_binned = pd.cut(
        acceptance_func_dfs['q2'], bins=q2_eval,
        labels=[f'e{num}' for num in range(len(q2_eval)-1)] # e0, e1, e2, ...
    )
    af_q2_eval = dict(tuple(acceptance_func_dfs.groupby(eval_binned)))

    ctl_inner_bins = [
        np.percentile(af_q2_eval[q2_bins]['costhetal'], ctl_percentile)
        for q2_bins in af_q2_eval
    ]

    ctl_inner_bins = np.asarray(ctl_inner_bins)
    ones_column = np.ones((ctl_inner_bins.shape[0], 1))
    ctl_bins = np.hstack((-ones_column, ctl_inner_bins, ones_column))

    return ctl_bins

def calc_ebins_cnt(df, q2_eval, ctl_bins):
    df_ebins = pd.cut(
        df['q2'], bins=q2_eval,
        labels=[f'e{num}' for num in range(len(q2_eval)-1)]
    )
    df_ebins = dict(tuple(df.groupby(df_ebins)))

    df_cnt = np.array([np.histogram(
        df_ebins[f'e{bin_no}']['costhetal'], bins=ctl_bins[bin_no])[0]
        for bin_no in range(len(q2_eval)-1)])

    return df_cnt

def rescale_q2(q2_array):
    return (q2_array-9.55)/9.45

def make_Leg(poly_degree):
    P = []
    for i in range(poly_degree):
        Leg_int_coeff = np.zeros(i+1)
        Leg_int_coeff[-1] = 1
        P.append(Legendre(Leg_int_coeff))
    return P

def acceptance_function(q2, ctl, params_dict):
    '''
    Continuous acceptance function.

    Parameters
    ----------
    q2 : int/float/1D array
        q² values to evaluate the acceptance function at
    ctl : int/float/1D array
        cos(θ_l) values to evaluate the acceptance function at
    params_dict : dict
        dictionary containing required parameter values (P and c)

    Returns
    -------
    2D array, with shape len(q2)xlen(ctl)
        where each row corresponds to an input q² value and each column to
        a cos(θ_l) value
    '''

    # extract required values from params_dict
    P = params_dict['P']
    c = params_dict['c']
    # we can also just use params_dict['i_range'], but c found using these
    # anyway, so we just use c's shape
    i_range = c.shape[0]
    j_range = c.shape[1]

    q2 = np.reshape(q2, (1, -1))
    ctl = np.reshape(ctl, (1, -1))

    P_i = np.array([P[i](q2) for i in range(i_range)])[...,None]
    P_j = np.array([P[j](ctl) for j in range(j_range)])

    func_val = (c.reshape(i_range, j_range, 1, 1) * (P_i*P_j)).sum((0, 1))
    return func_val

