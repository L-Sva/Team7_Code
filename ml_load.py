
from core import load_file, RAWFILES
import ml_tools
import pandas as pd

def get_train_test_for_all_peaking_bks(train_samples_limit=None):

    signal_dataset = load_file(RAWFILES.SIGNAL)

    peaking_bks_train = []
    peaking_bks_test = []
    for file in RAWFILES.peaking_bks:
        data = load_file(file)
        train, test = ml_tools.ml_prepare_train_test(data)
        peaking_bks_train.append(train)
        peaking_bks_test.append(test)

    peaking_bks_train = pd.concat(peaking_bks_train)
    peaking_bks_test = pd.concat(peaking_bks_test)

    sig_train, sig_test = ml_tools.ml_prepare_train_test(signal_dataset)

    for df in [peaking_bks_test, peaking_bks_train, sig_test, sig_train]:
        ml_tools.ml_strip_columns(df, inplace=True)

    train_data = ml_tools.ml_combine_signal_bk(sig_train[:train_samples_limit], peaking_bks_train[:train_samples_limit])
    test_data = ml_tools.ml_combine_signal_bk(sig_test, peaking_bks_test)

    return train_data, test_data

def get_train_validate_test_for_all_peaking_bks(train_samples_limit=None):

    signal_dataset = load_file(RAWFILES.SIGNAL)

    peaking_bks_train = []
    peaking_bks_validate = []
    peaking_bks_test = []
    for file in RAWFILES.peaking_bks:
        data = load_file(file)
        train, validate, test = ml_tools.ml_prepare_train_validate_test(data)
        peaking_bks_train.append(train)
        peaking_bks_validate.append(validate)
        peaking_bks_test.append(test)

    peaking_bks_train = pd.concat(peaking_bks_train)
    peaking_bks_validate = pd.concat(peaking_bks_validate)
    peaking_bks_test = pd.concat(peaking_bks_test)

    sig_train, sig_validate, sig_test = ml_tools.ml_prepare_train_validate_test(signal_dataset)

    for df in [peaking_bks_test, peaking_bks_train, peaking_bks_validate,
        sig_test, sig_train, sig_validate
    ]:
        ml_tools.ml_strip_columns(df, inplace=True)

    train_data = ml_tools.ml_combine_signal_bk(sig_train[:train_samples_limit], peaking_bks_train[:train_samples_limit])
    validate_data = ml_tools.ml_combine_signal_bk(sig_validate, peaking_bks_validate)
    test_data = ml_tools.ml_combine_signal_bk(sig_test, peaking_bks_test)

    return train_data, validate_data, test_data

def get_train_validate_test_for_background(bk_filename):
    """ Returns train test data for a specific peaking background vs. signal
    """
    signal_data = load_file(RAWFILES.SIGNAL)
    bk_data = load_file(bk_filename)

    sig_train, sig_validate, sig_test = ml_tools.ml_prepare_train_validate_test(signal_data)
    bks_train, bks_validate, bks_test = ml_tools.ml_prepare_train_validate_test(bk_data)

    for df in [sig_train, sig_validate, sig_test, bks_train, bks_validate, bks_test]:
        ml_tools.ml_strip_columns(df, inplace=True)

    train_data = ml_tools.ml_combine_signal_bk(sig_train, bks_train)
    validate_data = ml_tools.ml_combine_signal_bk(sig_validate, bks_validate)
    test_data = ml_tools.ml_combine_signal_bk(sig_test, bks_test)

    return train_data, validate_data, test_data