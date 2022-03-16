#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

from core import RAWFILES, load_file
from ES_functions.modifiedselectioncuts import (q2_resonances,
                                                selection_all_withoutres)
from ml_selector import remove_combinatorial_background
from Fitting_New.functions_new import calc_coeff_4d


if __name__ == '__main__':
    Path('../tmp/').mkdir(exist_ok=True)

    # create filtered acceptance_mc
    if not (
        Path('../tmp/filtered_acc_with_res.pkl').exists() and 
        Path('../tmp/filtered_acc_without_res.pkl').exists()):
        dataframe = load_file(RAWFILES.ACCEPTANCE)
        dataframe, _ = selection_all_withoutres(dataframe)
        dataframe_with_res, _ = remove_combinatorial_background(dataframe)
        dataframe_with_res.to_pickle('../tmp/filtered_acc_with_res.pkl')
        dataframe_without_res, _ = q2_resonances(dataframe_with_res)
        dataframe_without_res.to_pickle('../tmp/filtered_acc_without_res.pkl')
    
    dataframe_with_res = pd.read_pickle('../tmp/filtered_acc_with_res.pkl')
    dataframe_without_res = pd.read_pickle('../tmp/filtered_acc_without_res.pkl')
    
    c = calc_coeff_4d(dataframe_with_res)
    # print(c[0])
    # 0.25, 0.00192134, -0.0421559, 0.00075387, 0.00114201
    np.save('../tmp/coeff_4d.npy', c)

