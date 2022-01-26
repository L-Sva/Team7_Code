from core import load_file, RAWFILES
import ml_tools
import pandas as pd

def get_test_train_for_all_peaking_bks(train_samples_limit=None):

    signal_dataset = load_file(RAWFILES.SIGNAL)

    peaking_bks_train = []
    peaking_bks_test = []
    for file in RAWFILES.peaking_bks:
        data = load_file(file)
        train, test = ml_tools.ml_prepare_test_train(data)
        peaking_bks_train.append(train)
        peaking_bks_test.append(test)

    peaking_bks_train = pd.concat(peaking_bks_train)
    peaking_bks_test = pd.concat(peaking_bks_test)

    sig_train, sig_test = ml_tools.ml_prepare_test_train(signal_dataset)

    for df in [peaking_bks_test, peaking_bks_train, sig_test, sig_train]:
        ml_tools.ml_strip_columns(df, inplace=True)

    train_data = ml_tools.ml_combine_signal_bk(sig_train[:train_samples_limit], peaking_bks_train[:train_samples_limit])
    test_data = ml_tools.ml_combine_signal_bk(sig_test, peaking_bks_test)

    return train_data, test_data

def get_test_train_for_background(bk_filename):
    """ Returns train test data for a specific peaking background vs. signal
    """
    # load signal
    # load bk file
    # compose train_data, test_data
    return train_data, test_data