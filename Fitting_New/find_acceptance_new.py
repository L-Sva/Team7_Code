#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from core import RAWFILES, load_file
from ES_functions.modifiedselectioncuts import (q2_resonances,
                                                selection_all_withoutres)
from ml_selector import remove_combinatorial_background
from Fitting_New.functions_new import acceptance_function, calc_coeff


if __name__ == '__main__':
    Path('../tmp/').mkdir(exist_ok=True)

    # run this section if this is the first time computing acceptance function
    # filter acceptance_mc
    dataframe = load_file(RAWFILES.ACCEPTANCE)
    dataframe, _ = selection_all_withoutres(dataframe)
    dataframe_with_res, _ = remove_combinatorial_background(dataframe)
    dataframe_with_res.to_pickle('../tmp/filtered_acc_with_res.pkl')
    dataframe_without_res, _ = q2_resonances(dataframe_with_res)
    dataframe_without_res.to_pickle('../tmp/filtered_acc_without_res.pkl')
    c = calc_coeff(dataframe_with_res)
    np.save('../tmp/coeff.npy', c)

    # # run this section to avoid repeated calculation
    # dataframe_with_res = pd.read_pickle('../tmp/filtered_acc_with_res.pkl')
    # dataframe_without_res = pd.read_pickle('../tmp/filtered_acc_without_res.pkl')
    # c = np.load('../tmp/coeff.npy')

