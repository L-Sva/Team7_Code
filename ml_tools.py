from typing import Tuple
from core import RAWFILES, load_file
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

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

    # Drops any columns added during processing not specified to keep
    for name in dataframe:
        if (
            not (name in BASE_NAMES or name in accepted_column_names or name == 'category')
            or name in reject_column_names or name in columns_names_to_drop
        ):
            dataframe.drop(name, inplace=True, axis=1)

    return dataframe

def ml_train_model(training_data, model, **kwargs):
    """Trains a ML model. Requires that the parameter `training_data` contains a column named 'category'
    which will be the value the ML model is trained to predict; this should contain only integers,
    preferably only 0 or 1.
    """

    train_vars = training_data.drop('category',axis=1)
    model.fit(train_vars, training_data['category'].to_numpy(), **kwargs)
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
    signal_dataset = signal_dataset.copy()
    background_dataset = background_dataset.copy()
    signal_dataset.loc[:,'category'] = 1
    background_dataset.loc[:,'category'] = 0

    # combine
    dataset = pd.concat((signal_dataset, background_dataset))
    return dataset

def ml_get_model_sig_prob(testData, model):
    if 'category' in testData:
        test_vars = testData.drop('category',axis=1)
    return model.predict_proba(test_vars)[:,1]

def test_false_true_negative_positive(test_dataset, sig_prob, threshold) -> dict:
    # Jiayang

    signal = 0
    background = 0
    true_positive = 0
    false_negtive = 0
    false_positive = 0
    true_negative = 0

    x = test_dataset['category'].to_numpy()
    for i in range(len(x)):
        if (x[i] == 1) and (sig_prob[i] >= threshold): # signal + postive
            signal += 1
            true_positive += 1
        elif (x[i] == 1) and (sig_prob[i] < threshold): # signal + negative
            signal += 1
            false_negtive += 1
        elif (x[i] == 0) and (sig_prob[i] >= threshold): # background + postive
            background += 1
            false_positive +=1
        elif (x[i] == 0) and (sig_prob[i] < threshold): # background + negative
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


def roc_curve(model, test_dataset):
    # Jose
    pass

def plot_roc_curve():
    # Jose
    pass

def test_sb(test_dataset, sig_prob, threshold):
    # Jiayang

    output = test_false_true_negative_positive(test_dataset, sig_prob, threshold)

    S = output['signal'] * output['true-positive']
    B = output['background'] * output['false-positive']
    metric = S/np.sqrt(S+B)

    return metric
