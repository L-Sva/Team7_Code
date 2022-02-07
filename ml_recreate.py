
from typing import Tuple
from pandas import DataFrame
import scipy
import xgboost
from core import ensure_dir, load_file, RAWFILES
from histrogram_plots import generic_selector_plot, plot_hist_quantity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import ml_tools

def load_train_validate_test(file, test_size=0.2, validate_size=0.16):
    data = load_file(file)
    data = data.drop('year',axis=1)
    if test_size == 0:
        return data
    else:
        m = int(
            len(data) * (1-test_size)
        )
        if validate_size == 0:
            return data[0:m], data[m:]
        else:
            p = int(m * (1-validate_size))
            return data[0:p], data[p:m], data[m:]
        

def combine_signal_background(signal: Tuple[DataFrame],background: Tuple[DataFrame]):
    res = []
    for i, (s,b) in enumerate(zip(signal,background)):
        s = s.copy()
        b = b.copy()
        s.loc[:,'category'] = 1
        b.loc[:,'category'] = 0
        combine = pd.concat((s,b), ignore_index=True)
        if i >= 1:
            combine = pd.concat((b,s), ignore_index=True)
        combine = combine.sample(frac=1, axis=0, random_state=1)
        combine = combine.reset_index(drop=True)
        res.append(combine)
    return res

def concat_datasets(datasets):
    res = []
    for to_combine in zip(*datasets):
        combine = pd.concat(to_combine, ignore_index=True)
        res.append(combine)
    return res

def fit_new_model(train_data, **params):
    default_params = {
        'n_estimators': 400,
        'subsample': 1,
        'max_depth': 6,
        'learning_rate': 0.05,
        'gamma': 0,
        'reg_alpha': 1,
        'reg_lambda': 2,
    }
    for key, value in params.items():
        default_params[key] = value
    model = xgboost.XGBClassifier(use_label_encoder=False,**default_params)
    model.fit(
        train_data.drop('category',axis=1).values,
        train_data['category'].to_numpy().astype('int'),
        eval_metric='logloss'
    )
    return model

def load_model_file(path):
    params = {
        'n_estimators': 400,
        'subsample': 1,
        'max_depth': 6,
        'learning_rate': 0.05,
        'gamma': 0,
        'reg_alpha': 1,
        'reg_lambda': 2,
    }
    model = xgboost.XGBClassifier(use_label_encoder=False,**params)
    model.load_model(path)
    return model

def select(data, model, thresh):
    if 'category' in data:
        data = data.drop('category',axis=1)
    sig_prob = model.predict_proba(data.values)[:,1]
    accept = sig_prob > thresh
    s = data[accept]
    ns = data[~accept]
    return s, ns

def make_selector(model, thresh):
    def generated_selector(data):
        return select(data, model, thresh)
    return generated_selector

def plot_sb(data, sig_prob):
    thresh_list = np.linspace(0.1, 1, 600)
    sb = [
        ml_tools.test_sb(data,sig_prob, t) for t in thresh_list
    ]
    plt.plot(thresh_list,sb)

def optimize_threshold(validate_dataset, sig_prob):
    def target_func(x):
        return 400-ml_tools.test_sb(validate_dataset, sig_prob, x)

    optimize_result = scipy.optimize.dual_annealing(target_func,bounds=((0,1),),x0=[0.95])
    thresh = optimize_result.x[0]
    return thresh

def predict_prob(data, model):
    return model.predict_proba(data.drop('category',axis=1).values)[:,1]

if __name__ == '__main__':

    signal = load_train_validate_test(RAWFILES.SIGNAL, validate_size=0)
    background = load_train_validate_test(RAWFILES.JPSI, validate_size=0)
    background = concat_datasets([load_train_validate_test(file) for file in RAWFILES.peaking_bks])
    train, test = combine_signal_background(signal, background)


    MODEL_PATH = os.path.join('examples_save','0008_peak.model')

    if not os.path.exists(MODEL_PATH):
        model = fit_new_model(train[:30000])
        model.save_model(MODEL_PATH)
    else:
        model = load_model_file(MODEL_PATH)

    sig_prob = predict_prob(test, model)
    thresh = optimize_threshold(test, sig_prob)

    print(ml_tools.test_false_true_negative_positive(test, sig_prob, thresh))
    plot_sb(test, sig_prob)
    plt.close()

    selector = make_selector(model, thresh)
    s, ns = selector(background[0])
    print(len(s), len(ns))

    IMAGE_OUTPUT_DIR = 'data_ml_selectors_histograms'

    ensure_dir(IMAGE_OUTPUT_DIR)

    total = load_train_validate_test(RAWFILES.TOTAL_DATASET, test_size = 0, validate_size= 0)
    signal_all = load_train_validate_test(RAWFILES.SIGNAL, test_size = 0, validate_size= 0)
    s, ns = selector(total)
    for column in total:
        bins, h = plot_hist_quantity(total, column, label='Total dataset', bins=150)
        plot_hist_quantity(s, column, label='ML - Signal', bins = bins)
        plot_hist_quantity(ns, column, label='ML - Background', bins=bins)
        plot_hist_quantity(signal_all, column, label='Simulated signal', bins=bins)
        plt.legend()

        plt.savefig(
            os.path.join(IMAGE_OUTPUT_DIR,f'{column}.png')
        )
        plt.close()

    DIR ='ml_hist_individual_bks'
    OUTPUT_PLOTS = False
    ensure_dir(DIR)

    for file in RAWFILES.peaking_bks:
        IMG_DIR = os.path.join(DIR, file[:-4])
        _, test = load_train_validate_test(file, validate_size=0)
        s, ns = selector(test)
        num = len(test)
        print(f'{file} | accepted: {len(s)} ({len(s)/num}), rejected {len(ns)} ({len(ns)/num})')
        if OUTPUT_PLOTS:
            cols = [col for col in test]
            N = len(cols)
            i = 1
            for col in test:
                generic_selector_plot(test, s, ns, col, 150, False)
                plt.title(f'ML peaking bk removal on {file[:-4]}')
                plt.savefig(os.path.join(IMG_DIR, f'{file[:-4]}_{col}.png'))
                plt.close()
                print(f'Outputing plots for {file} | {i} / {N}', end='\r')
                i += 1
            print('    Done: ')