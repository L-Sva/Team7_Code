from typing import Tuple
from core import RAWFILES, load_file
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_NAMES = [name for name in load_file(RAWFILES.SIGNAL)]

def ml_strip_columns(dataframe, accepted_column_names: Tuple[str, ...]=()) -> pd.DataFrame:
    """Strips columns which contain information we don't want to pass to the ML model"""

    # Drops 'year' and 'B0_ID' columns
    columns_names_to_drop = ('year','B0_ID')
    for name in columns_names_to_drop:
        dataframe = dataframe.drop(name)
    
    # Drops any columns added during processing not specified to keep
    for name in dataframe:
        if not name in BASE_NAMES or name in accepted_column_names or name == 'category':
            dataframe = dataframe.drop(name)

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
    return {
        'true-positive': 0.8,
        'false-positive': 1,
        'true-negative': 0.1,
        'false-negative': 0.2
    }

def roc_curve(model, test_dataset):
    # Jose
    pass

def plot_roc_curve():
    # Jose
    pass

def test_sb(model, test_dataset, threshold):
    # Jiayang
    # Author should find a better name
    pass


