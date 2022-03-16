#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import pandas as pd
import vegas

from Fitting_New.functions_new import (acceptance_function,
                                       acceptance_function_4d, q2_binned)


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


def save_reduced():
    @vegas.batchintegrand
    def term_1(x, coeff):
        return acceptance_function(*x, coeff)

    @vegas.batchintegrand
    def term_2(x, coeff):
        return acceptance_function(*x, coeff) * (2 * x[1]**2 - 1)

    @vegas.batchintegrand
    def term_3(x, coeff):
        return acceptance_function(*x, coeff) * x[1]

    terms = [term_1, term_2, term_3]

    coeff = np.load('../tmp/coeff.npy')

    def normalization_1D(coeff, bin_no):
        norm = vegas.Integrator(
        [ # integral limits for q2, ctl
            [bin_dic[bin_no][0], bin_dic[bin_no][1]],
            [-1, 1]
        ])

        I = []

        for term in terms:
            norm(partial(term, coeff=coeff), nitn=10, neval=1e4)
            result = norm(partial(term, coeff=coeff), nitn=10, neval=1e4)
            I.append(result[0].mean)

        return np.array(I)

    delta_normed = [normalization_1D(coeff, bin_no)
    for bin_no in range(10)]
    np.save('../tmp/delta_normed.npy', delta_normed) # save to be called later

    print('Done saving reduced delta_normed.')

def save_8d():
    @vegas.batchintegrand
    def term_1_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * (1 - x[2]**2)

    @vegas.batchintegrand
    def term_2_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * x[2]**2

    @vegas.batchintegrand
    def term_3_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * (1 - x[2]**2) * \
            (2*x[1]**2 - 1)

    @vegas.batchintegrand
    def term_4_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * x[2]**2 * (2*x[1]**2 - 1)

    @vegas.batchintegrand
    def term_5_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * (1 - x[2]**2) * \
        (1 - x[1]**2) * np.cos(2 * x[3])

    @vegas.batchintegrand
    def term_6_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * 2 * x[2] * \
        np.sqrt(1 - x[2]**2) * 2 * x[1] * np.sqrt(1- x[1]**2) * np.cos(x[3])

    @vegas.batchintegrand
    def term_7_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * 2 * x[2] * \
            np.sqrt(1 - x[2]**2) * np.sqrt(1 - x[1]**2) * np.cos(x[3])

    @vegas.batchintegrand
    def term_8_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * (1 - x[2]**2) * x[1]

    @vegas.batchintegrand
    def term_9_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * 2 * x[2] * \
        np.sqrt(1 - x[2]**2) * np.sqrt(1 - x[1]**2) * np.sin(x[3])

    @vegas.batchintegrand
    def term_10_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * 2 * x[2] * \
        np.sqrt(1 - x[2]**2) * 2 * x[1] * np.sqrt(1 - x[1]**2)  * np.sin(x[3])

    @vegas.batchintegrand
    def term_11_full(x, coeff):
        return acceptance_function_4d(*x, coeff) * (1 - x[2]**2) * \
        (1 - x[1]**2) * np.sin(2 * x[3])

    terms = [term_1_full, term_2_full, term_3_full, term_4_full, term_5_full,
    term_6_full, term_7_full, term_8_full, term_9_full, term_10_full,
    term_11_full]

    coeff_4d = np.load('../tmp/coeff_4d.npy')

    def normalization(coeff, bin_no):
        norm = vegas.Integrator(
            [   # integral limits for q2, ctl, ctk, phi
                [bin_dic[bin_no][0], bin_dic[bin_no][1]],
                [-1, 1],
                [-1, 1],
                [-np.pi, np.pi]
            ])

        I = []

        for term in terms:
            norm(partial(term, coeff=coeff), nitn=10, neval=1e3)
            result = norm(partial(term, coeff=coeff), nitn=10, neval=1e3)
            I.append(result[0].mean)

        return I

    delta_normed_8d = np.array([normalization(coeff_4d, bin_no)
    for bin_no in range(10)])
    print(repr(delta_normed_8d))

    np.save('../tmp/delta_normed_8d.npy', delta_normed_8d)

    print('Done saving all delta_normed_8d.')


# save_reduced()
# save_8d()

