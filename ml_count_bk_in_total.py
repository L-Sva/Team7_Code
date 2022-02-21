import json
from operator import mod
import os

from core import RAWFILES, ensure_dir, load_file
from ml_recreate import (
    combine_signal_background, concat_datasets,
    file_cached_model_fit, load_model_file, load_train_validate_test, make_selector, optimize_threshold, 
    predict_prob, select, split_train_validate_test
)
from ml_tools import test_false_true_negative_positive
from ml_combinatorial_extraction import B0_MM_selector
from ml_selector import remove_all_bk

TRAIN = False

MODELS_PATH = os.path.join('ml_models','selecting_background')
THRESHOLDS_PATH = os.path.join(MODELS_PATH,f'thresholds.json')
ensure_dir(MODELS_PATH)

def create_bk_select_model(bk_sig_file, all_sim_files, thresholds):
    total = load_file(RAWFILES.TOTAL_DATASET)
    sig = load_train_validate_test(bk_sig_file, validate_size=0)
    bk = concat_datasets([load_train_validate_test(f, validate_size=0) for f in all_sim_files if (
        f != bk_sig_file
    )])
    train, test = combine_signal_background(sig, bk)
    model_name = bk_sig_file.split('.')[0]
    _train_stats_model(total, train, test, model_name, thresholds)

def comb_only_remover(all_sim_files, thresholds):
    total = load_file(RAWFILES.TOTAL_DATASET)
    sig = concat_datasets([load_train_validate_test(f) for f in all_sim_files])
    bk_data, _ = B0_MM_selector(load_file(RAWFILES.TOTAL_DATASET), 5400)
    bk_data = split_train_validate_test(bk_data, validate_size=0)
    train, test = combine_signal_background(sig, bk_data)
    model_name = 'combinatorial_only_removal'
    _train_stats_model(total,train,test, model_name, thresholds)

def _train_stats_model(total, train, test, model_name, thresholds):
    print(f'Training / loading {model_name} model')
    model = file_cached_model_fit(
        os.path.join(MODELS_PATH,f'{model_name}.model'),
        train
    )
    sig_prob = predict_prob(test, model)
    thresh = optimize_threshold(test, sig_prob)
    thresholds[model_name] = thresh
    print(test_false_true_negative_positive(test, sig_prob, thresh))
    s, ns = select(total, model, thresh)
    print(f'{model_name} | accepted: {len(s)}, reject: {len(ns)}')

peakingbks_files = RAWFILES.peaking_bks
all_sim_files = RAWFILES.peaking_bks + [RAWFILES.SIGNAL,]
THRESHOLDS = {}
if os.path.exists(THRESHOLDS_PATH):
    with open(THRESHOLDS_PATH, 'r') as file:
        THRESHOLDS = json.load(file)

if TRAIN:
    for bk_sig_file in peakingbks_files:
        create_bk_select_model(bk_sig_file, all_sim_files, THRESHOLDS)
    comb_only_remover(all_sim_files, THRESHOLDS)
    with open(THRESHOLDS_PATH, 'w') as file:
        json.dump(THRESHOLDS, file)

total = load_file(RAWFILES.TOTAL_DATASET)
comb_model_name = 'combinatorial_only_removal'

def make_selector_from_name(model_name):
    return make_selector(
        load_model_file(os.path.join(MODELS_PATH,f'{model_name}.model')),
        THRESHOLDS[model_name]
    )

comb_selector = make_selector_from_name(comb_model_name)
non_comb_bk, comb_bk = comb_selector(total)
print(f'{comb_model_name} | accepted {len(non_comb_bk)}, rejected {len(comb_bk)}')

ALL_ACCEPTED_PROPORTION = {}
FPR = {}

def info(model_name, selector, data=non_comb_bk):
    s, ns = selector(data)
    print(f'{model_name} | accepted {len(s)}, rejected {len(ns)}')
    ALL_ACCEPTED_PROPORTION[model_name] = len(s) / len(data)

for file in peakingbks_files:
    model_name = file.split('.')[0]
    _, fpr_test = load_train_validate_test(file, validate_size=0)
    selector = make_selector_from_name(model_name)
    num = len(fpr_test)
    s, ns = remove_all_bk(fpr_test)
    fpr = len(s) / num
    FPR[model_name] = fpr
    info(model_name, selector)

print('Estimated number of remaining bk events')
for key in ALL_ACCEPTED_PROPORTION:
    print(f'{key}| {ALL_ACCEPTED_PROPORTION[key] }|{FPR[key]}|{ALL_ACCEPTED_PROPORTION[key] * FPR[key] * len(non_comb_bk)}')

print('Sum of proportions')
print(sum([val for key, val in ALL_ACCEPTED_PROPORTION.items()]))