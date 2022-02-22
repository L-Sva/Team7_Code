import json
import os
import pandas as pd

from core import RAWFILES, ensure_dir, load_file
from ml_main import (
    combine_signal_background, concat_datasets,
    file_cached_model_fit, load_model_file, load_train_validate_test, make_selector, optimize_threshold, 
    predict_prob, split_train_validate_test
)
from ml_tools import test_false_true_negative_positive
from ml_combinatorial_extraction import B0_MM_selector
from ml_selector import remove_all_bk

MODELS_PATH = os.path.join('ml_models','selecting_background')
DETAILS_PATH = os.path.join(MODELS_PATH,f'details.json')
ensure_dir(MODELS_PATH)

def create_bk_select_model(bk_sig_file, all_sim_files, model_details):
    model_name = bk_sig_file.split('.')[0]
    if not model_name in model_details:
        total = load_file(RAWFILES.TOTAL_DATASET)
        sig = load_train_validate_test(bk_sig_file, validate_size=0)
        bk = concat_datasets([load_train_validate_test(f, validate_size=0) for f in all_sim_files if (
            f != bk_sig_file
        )])
        train, test = combine_signal_background(sig, bk)
        _train_stats_model(total, train, test, model_name, model_details)

def comb_only_remover(all_sim_files, model_details):
    model_name = 'combinatorial_only_removal'
    if not model_name in model_details:
        total = load_file(RAWFILES.TOTAL_DATASET)
        sig = concat_datasets([load_train_validate_test(f) for f in all_sim_files])
        bk_data, _ = B0_MM_selector(load_file(RAWFILES.TOTAL_DATASET), 5400)
        bk_data = split_train_validate_test(bk_data, validate_size=0)
        train, test = combine_signal_background(sig, bk_data)
        _train_stats_model(total,train,test, model_name, model_details)

def _train_stats_model(total, train, test, model_name, model_details):
    print(f'Training / loading {model_name} model')
    model = file_cached_model_fit(
        os.path.join(MODELS_PATH,f'{model_name}.model'),
        train
    )
    sig_prob = predict_prob(test, model)
    thresh = optimize_threshold(test, sig_prob)
    test_result = test_false_true_negative_positive(test, sig_prob, thresh)
    print(test_result)
    model_details[model_name] = {
        'threshold': thresh,
        'tpr': test_result['true-positive'],
        'fpr': test_result['false-positive']
    }

peakingbks_files = RAWFILES.peaking_bks
all_sim_files = RAWFILES.peaking_bks + [RAWFILES.SIGNAL,]

DETAILS = {}
if os.path.exists(DETAILS_PATH):
    with open(DETAILS_PATH, 'r') as file:
        DETAILS = json.load(file)

for file in all_sim_files:
    create_bk_select_model(file, all_sim_files, DETAILS)
create_bk_select_model(RAWFILES.SIGNAL, all_sim_files, DETAILS)
comb_only_remover(all_sim_files, DETAILS)
with open(DETAILS_PATH, 'w') as file:
    json.dump(DETAILS, file)

total = load_file(RAWFILES.TOTAL_DATASET)
comb_model_name = 'combinatorial_only_removal'

def make_selector_from_name(model_name):
    return make_selector(
        load_model_file(os.path.join(MODELS_PATH,f'{model_name}.model')),
        DETAILS[model_name]['threshold']
    )

comb_selector = make_selector_from_name(comb_model_name)
non_comb_bk, comb_bk = comb_selector(total)
print(f'{comb_model_name} | accepted {len(non_comb_bk)}, rejected {len(comb_bk)}')

proportion_accepted_bk_estimation = {}
background_removal_fpr = {}

def info(model_name, selector, data=non_comb_bk):
    s, ns = selector(data)
    print(f'{model_name} | accepted {len(s)}, rejected {len(ns)}')
    proportion_accepted_bk_estimation[model_name] = len(s) / len(data)

for file in all_sim_files:
    model_name = file.split('.')[0]
    _, fpr_test = load_train_validate_test(file, validate_size=0)
    selector = make_selector_from_name(model_name)
    num = len(fpr_test)
    s, ns = remove_all_bk(fpr_test)
    fpr = len(s) / num
    background_removal_fpr[model_name] = fpr
    info(model_name, selector)

results = []
total_num = len(non_comb_bk)

for key in proportion_accepted_bk_estimation:
    acc = proportion_accepted_bk_estimation[key]
    tpr = DETAILS[key]['tpr']
    fpr = DETAILS[key]['fpr']
    S_proportion_estimate = ( fpr * (1-acc) - (1-fpr) * acc ) / (fpr - tpr)
    results.append([
        key,
        background_removal_fpr[key],
        S_proportion_estimate,
        S_proportion_estimate * background_removal_fpr[key] * total_num
    ])

results = pd.DataFrame(results,columns=['Decay_name','Rate_selected_bk_remover','Estimate_prop_total','Estimate_N_accepted'])

print()
print(results)

print()
print('Sum:')
print(results.sum(axis=0))

