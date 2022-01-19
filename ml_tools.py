from typing import Tuple
from core import RAWFILES, load_file
import pandas as pd

BASE_NAMES = [name for name in load_file(RAWFILES.SIGNAL)]

def ml_strip_columns(dataframe, accepted_column_names: Tuple=()):
    """Strips columns which contain information we don't want to pass to the ML model"""

    # Drops 'year' and 'B0_ID' columns
    columns_names_to_drop = ('year','B0_ID')
    for name in columns_names_to_drop:
        dataframe = dataframe.drop(name)
    
    # Drops any columns added during processing not specified to keep
    for name in dataframe:
        if not name in BASE_NAMES or name in accepted_column_names:
            dataframe = dataframe.drop(name)

    return dataframe

def ml_prepare_test_train(dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Takes a dataset and splits it into test and train datasets"""
    pass

def ml_train_model(training_data, model):
    """Trains a ML model. Requires that the parameter `training_data` contains a column named 'category'
    which will be the value the ML model is trained to predict; this should contain only integers,
    preferably only 0 or 1.
    """

    train_vars = training_data.drop('category')
    model.fit(train_vars, training_data['category'])
    return model