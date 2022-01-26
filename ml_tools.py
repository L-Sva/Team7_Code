from typing import Tuple
from core import RAWFILES, load_file
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier

BASE_NAMES = [name for name in load_file(RAWFILES.SIGNAL)]

def ml_strip_columns(dataframe,
    accepted_column_names: Tuple[str, ...]=(),
    reject_column_names: Tuple[str, ...]=(),
    inplace=False
) -> pd.DataFrame:
    """Strips columns which contain information we don't want to pass to the ML model"""

    if not inplace:
        dataframe = dataframe.copy()

    # Drops 'year' and 'B0_ID' columns
    columns_names_to_drop = ('year','B0_ID')
    for name in columns_names_to_drop:
        dataframe = dataframe.drop(name)

    # Drops any columns added during processing not specified to keep
    for name in dataframe:
        if (
            not (name in BASE_NAMES or name in accepted_column_names or name == 'category')
            or name in reject_column_names
        ):
            dataframe.drop(name, inplace=True)

    return dataframe

def ml_train_model(training_data, model):
    """Trains a ML model. Requires that the parameter `training_data` contains a column named 'category'
    which will be the value the ML model is trained to predict; this should contain only integers,
    preferably only 0 or 1.
    """

    train_vars = training_data.drop('category')
    model.fit(train_vars, training_data['category'])
    return model

def ml_prepare_test_train(dataset, randomiser_seed = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Takes a dataset and splits it into test and train datasets"""
    # Marek
    train, test = train_test_split(dataset, test_size = 0.2, random_state=randomiser_seed)
    return train, test


def ml_combine_signal_bk(signal_dataset, background_dataset):
    """Combines signal and background dataset, adding category labels
    """
    # Marek
    signal_dataset['category'] = 1
    background_dataset['category'] = 1

    # combine
    dataset = pd.concat((signal_dataset, background_dataset))
    return dataset


def test_false_true_negative_positive(model, test_dataset, threshold) -> dict:
    # Jiayang

    sig_prob = model # signal probability from bdt model


    signal = 0
    background = 0
    true_positive = 0
    false_negtive = 0
    false_positive = 0
    true_negative = 0

    for i in range(len(test_dataset['catagory'])):
        if (test_dataset['catagory'][i] == 1) and (sig_prob[i] >= threshold): # signal + postive
            signal += 1
            true_positive += 1
        elif (test_dataset['catagory'][i] == 1) and (sig_prob[i] < threshold): # signal + negative
            signal += 1
            false_negtive += 1
        elif (test_dataset['catagory'][i] == 0) and (sig_prob[i] >= threshold): # background + postive
            background += 1
            false_positive +=1
        elif (test_dataset['catagory'][i] == 0) and (sig_prob[i] < threshold): # background + negative
            background += 1
            true_negative +=1

    # sanity check
    # total = true_positive + false_negtive + false_positive + true_negative
    # print('total counted:', total, (signal+background))
    # print('total candidates:', len(test_dataset['catagory']))

    # rates
    tpr = true_positive / signal
    fpr = false_positive / background

    fnr = false_negtive / signal
    tnr = true_negative / background

    return {
        'true-positive': tpr,
        'false-positive': fpr,
        'true-negative': tnr,
        'false-negative': fnr,
        'signal': signal,
        'background': background
    }


def roc_function(sp, test_data):
    # Jose
    '''
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

    fpr, tpr, cut_values = roc_curve(test_data['category'], sp)
    area = auc(fpr, tpr)
    
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

def test_sb(model, test_dataset, threshold):
    # Jiayang

    output = test_false_true_negative_positive(model, test_dataset, threshold)

    S = output['signal'] * output['true-positive']
    B = output['background'] * output['false-positive']
    metric = S/np.sqrt(S+B)

    return metric
