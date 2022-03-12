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
from Fitting_New.functions_new import (acceptance_function,
log_likelihood_S, q2_binned)


if __name__ == '__main__':
    '''
    # fist run to generate the files
    dataframe = load_file(RAWFILES.TOTAL_DATASET)
    dataframe, _ = selection_all(dataframe)
    dataframe, _ = remove_combinatorial_background(dataframe)
    dataframe.to_pickle('../tmp/filtered_total_dataset.pkl')
    '''

    coeff = np.load('../tmp/coeff.npy')

    # read file to avoid recalculation
    dataframe = pd.read_pickle('../tmp/filtered_total_dataset.pkl')
    bins = q2_binned(dataframe)

    bins_log_likelihood_S = partial(log_likelihood_S, bins, coeff)

    bins_log_likelihood_S.errordef = Minuit.LIKELIHOOD

    results = []
    errors = []

    starting_point = [
        0.711290, 0.122155, -0.024751, -0.224204, 
        -0.337140, -0.013383, -0.005062,-0.000706
    ]
    m = Minuit(bins_log_likelihood_S, *starting_point, 1)

    # fixing the bin number as we don't want to optimize it
    m.fixed['_bin'] = True
    m.limits=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), 
              (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), None)
    # m.migrad()
    results.append(np.array(m.values))
    # errors.append(np.array(m.errors))
    #m.fmin
    #m.params

    print(results, errors)

