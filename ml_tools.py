from typing import Tuple
from core import RAWFILES, load_file

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