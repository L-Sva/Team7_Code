from typing import Tuple

from pandas import DataFrame
import xgboost
from core import load_file, RAWFILES
from histrogram_plots import plot_hist_quantity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import ml_tools

def load_train_test(file, test_size=0.2):
    data = load_file(file)
    data = data.drop('year',axis=1)
    m = int(
        len(data) * (1-test_size)
    )
    return data[0:m], data[m:]

def combine_datasets(signal: Tuple[DataFrame],background: Tuple[DataFrame]):
    res = []
    for i, (s,b) in enumerate(zip(signal,background)):
        s = s.copy()
        b = b.copy()
        s.loc[:,'category'] = 1
        b.loc[:,'category'] = 0
        combine = pd.concat((s,b), ignore_index=True)
        if i == 1:
            combine = pd.concat((b,s), ignore_index=True)
        combine = combine.sample(frac=1, axis=0, random_state=1)
        combine = combine.reset_index(drop=True)
        res.append(combine)
    return res

def fit_new_model(train_data):
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
    model.fit(
        train_data.drop('category',axis=1).values,
        train_data['category'].to_numpy().astype('int'),
        eval_metric='logloss'
    )
    return model


signal = load_train_test(RAWFILES.SIGNAL)
background = load_train_test(RAWFILES.JPSI)

combine = combine_datasets(signal, background)

model = fit_new_model(combine[0][:30000])
model.save_model(os.path.join('examples_save','0006_peak.model'))
sig_prob = model.predict_proba(combine[1].drop('category',axis=1).values)[:,1]

print(ml_tools.test_false_true_negative_positive(combine[1], sig_prob, 0.95))
thresh = np.linspace(0.1, 1, 600)
sb = [
    ml_tools.test_sb(combine[1],sig_prob, t) for t in thresh
]
plt.plot(thresh,sb)
plt.figure()

bins, h = plot_hist_quantity(signal[1], 'q2')
plot_hist_quantity(background[1], 'q2',bins=bins)
plot_hist_quantity(combine[1][combine[1]['category'] == 0],'q2',bins)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(combine[1].head())

# print(sig_prob)

plt.show()