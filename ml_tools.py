from typing import Tuple
from core import RAWFILES, load_file
import pandas as pd
#from sklearn.model_selection import train_test_split as _train_test_split
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import os, time

BASE_NAMES = [name for name in load_file(RAWFILES.SIGNAL)]

def ml_strip_columns(dataframe,
    accepted_column_names: Tuple[str, ...]=(),
    reject_column_names: Tuple[str, ...]=()
) -> pd.DataFrame:
    """Strips columns which contain information we don't want to pass to the ML model"""

    dataframe = dataframe.copy()

    # Drops 'year' and 'B0_ID' columns
    columns_names_to_drop = ('year','B0_ID')

    # Drops any columns added during processing not specified to keep
    for name in dataframe:
        if (
            not (name in BASE_NAMES or name in accepted_column_names or name == 'category')
            or name in reject_column_names or name in columns_names_to_drop
        ):
            dataframe.drop(name, inplace=True, axis=1)

    return dataframe

def test_false_true_negative_positive(test_dataset, sig_prob, threshold) -> dict:
    # Jiayang

    x = test_dataset['category'].to_numpy()

    x_mask_0 = x == 0
    x_mask_1 = x == 1
    prb_mask_pos = sig_prob >= threshold
    prb_mask_neg = sig_prob < threshold

    signal = np.count_nonzero(x_mask_1)
    background = np.count_nonzero(x_mask_0)
    true_positive =  np.count_nonzero(np.logical_and(x_mask_1, prb_mask_pos))
    false_negative = np.count_nonzero(np.logical_and(x_mask_1, prb_mask_neg))
    false_positive = np.count_nonzero(np.logical_and(x_mask_0, prb_mask_pos))
    true_negative =  np.count_nonzero(np.logical_and(x_mask_0, prb_mask_neg))
    
    # rates
    tpr = true_positive / signal
    fpr = false_positive / background

    fnr = false_negative / signal
    tnr = true_negative / background

    return {
        'true-positive': tpr,
        'false-positive': fpr,
        'true-negative': tnr,
        'false-negative': fnr,
        'signal': signal,
        'background': background,
        'n-signal-accept': signal * tpr,
        'n-background-accept': background * fpr,
    }


def roc_curve(model, test_data):
    # Jose
    '''
    Test data needs to be in pandas dataframe format.
    Implement the following model before this function:
        model = XGBClassifier()
        model.fit(training_data[training_columns], training_data['category'])
        sp = model.predict_proba(test_data[training_columns])[:,1]
        model.predict_proba(test_data[training_columns])
    This returns an array of N_samples by N_classes.
    The first column is the probability that the candiate is category 0 (background).
    The second column (sp) is the probability that the candidate is category 1 (signal).

    The Receiver Operating Characteristic curve given by this function shows the efficiency of the classifier
    on signal (true positive rate, tpr) against the inefficiency of removing background (false positive
    rate, fpr). Each point on this curve corresponds to a cut value threshold.
    '''

    sp = ml_get_model_sig_prob(test_data, model)
    fpr, tpr, cut_values = metrics.roc_curve(test_data['category'], sp)
    area = metrics.auc(fpr, tpr)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'cut_values': cut_values,
        'area': area
    }

def plot_roc_curve(fpr, tpr, area):
    # Jose

    plt.plot([0, 1], [0, 1], color='deepskyblue', linestyle='--', label='Random guess')
    plt.plot(fpr, tpr, color='darkblue', label=f'ROC curve (area = {area:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()

def test_sb(test_dataset, sig_prob, threshold):
    # Jiayang

    output = test_false_true_negative_positive(test_dataset, sig_prob, threshold)

    S = output['signal'] * output['true-positive']
    B = output['background'] * output['false-positive']
    if S+B == 0:
        return 0
    metric = S/np.sqrt(S+B)
    return metric


def bayesian_nextpoint(function, pbounds, random_state=1, **util_args):
    """
    Suggestion:
        Not to use this, but use the bayesian_optimisation function below.
        bc it does not perform optimisation continuously.

    input:
        random_state: int, default = 1
            can be an integer for consistent outputs, or None for random outputs

        util_args: dict
            tweak this (or random_state) to get different params each time,
            for example, util_args = {'kind':"ucb", 'kappa':2.5,'xi':0.0}

    output:
        next_point : dict
            a set of params within pbounds, for example: {'x': 123, 'y': 123}
    """
    optimizer = BayesianOptimization(function, pbounds, verbose=2, random_state=random_state)

    utility = UtilityFunction(**util_args)
    next_point = optimizer.suggest(utility)
    print("next_point:", next_point)

    return next_point

def bayesian_optimisation(function, 
    pbounds, log_folder, bool_load_logs = True, explore_runs = 2, exploit_runs = 1):
    """
    runs function to find optimal parameters

    output:
        result : dict
            for example, {'target': 123, 'params': {'x': 123, 'y': 123}}
            where target = function(params)
    """
    print('====== start bayesian optimisation ======')
    
    optimizer = BayesianOptimization(function, pbounds, verbose=2, random_state=1,)
    if bool_load_logs:
        log_folder_files = os.listdir(log_folder)
        logs=[
            os.path.join(log_folder, f) for f in log_folder_files if (
                f[0:5] == 'logs_' and f[-5:] == '.json'
            )
        ]
        load_logs(optimizer, logs=logs)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logger = JSONLogger(path=os.path.join(log_folder, f'logs_{timestr}'))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    if exploit_runs > 0 or exploit_runs > 0:
        optimizer.maximize(init_points = explore_runs, n_iter = exploit_runs,)
    print('====== end bayesian optimisation ======')

    return optimizer
