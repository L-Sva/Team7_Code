import os
from core import RAWFILES, ensure_dir, load_file
from ml_recreate import (
    combine_signal_background, concat_datasets,
    file_cached_model_fit, load_train_validate_test, optimize_threshold, 
    predict_prob, select, split_train_validate_test
)
from ml_tools import test_false_true_negative_positive
from ml_combinatorial_extraction import B0_MM_selector

MODELS_PATH = os.path.join('ml_models','selecting_background')
ensure_dir(MODELS_PATH)

def create_bk_select_model(bk_sig_file, all_sim_files):
    total = load_file(RAWFILES.TOTAL_DATASET)
    sig = load_train_validate_test(bk_sig_file, validate_size=0)
    bk = concat_datasets([load_train_validate_test(f, validate_size=0) for f in all_sim_files if (
        f != bk_sig_file
    )])
    train, test = combine_signal_background(sig, bk)
    model_name = bk_sig_file.split('.')[0]
    _train_stats_model(total, train, test, model_name)

def comb_only_remover(all_sim_files):
    total = load_file(RAWFILES.TOTAL_DATASET)
    sig = concat_datasets([load_train_validate_test(f) for f in all_sim_files])
    bk_data, _ = B0_MM_selector(load_file(RAWFILES.TOTAL_DATASET), 5400)
    bk_data = split_train_validate_test(bk_data, validate_size=0)
    train, test = combine_signal_background(sig, bk_data)
    model_name = 'combinatorial_only_removal'
    _train_stats_model(total,train,test, model_name)

def _train_stats_model(total, train, test, model_name):
    print(f'Training / loading {model_name} model')
    model = file_cached_model_fit(
        os.path.join(MODELS_PATH,f'{model_name}.model'),
        train
    )
    sig_prob = predict_prob(test, model)
    thresh = optimize_threshold(test, sig_prob)
    print(test_false_true_negative_positive(test, sig_prob, thresh))
    s, ns = select(total, model, thresh)
    print(f'{model_name} | accepted: {len(s)}, reject: {len(ns)}')

peakingbks_files = RAWFILES.peaking_bks
all_sim_files = RAWFILES.peaking_bks + [RAWFILES.SIGNAL,]

for bk_sig_file in peakingbks_files:
    create_bk_select_model(bk_sig_file, all_sim_files)

comb_only_remover(all_sim_files)