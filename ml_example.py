
from core import load_file, RAWFILES
import ml_tools
import pandas as pd
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import os

total_dataset = load_file(RAWFILES.TOTAL_DATASET)

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

train_data = ml_tools.ml_combine_signal_bk(sig_train[:10000], peaking_bks_train[:10000])
test_data = ml_tools.ml_combine_signal_bk(sig_test, peaking_bks_test)

xgboost.set_config(verbosity=2)
xge_model = xgboost.XGBClassifier(
    max_depth=4,
    num_boost_round = 10
)


print('Starting model training')
ml_tools.ml_train_model(train_data, xge_model, 
    eval_metric='logloss',
)
print('Completed model training')

xge_model.save_model(os.path.join('examples_save','0001_peaking.model'))

sig_prob = ml_tools.ml_get_model_sig_prob(test_data, xge_model) # signal probability from bdt model

threshold_list = np.sort(
    np.concatenate((np.linspace(0.8,1,20), np.linspace(0.97,0.995, 20)))
)
sb_list = []
for thresh in threshold_list:
    sb_list.append(ml_tools.test_sb(test_data, sig_prob, thresh))

plt.plot(threshold_list, sb_list)
plt.xlabel('Cut value')
plt.ylabel('$S / \\sqrt{S + B}$')
plt.show()

print(ml_tools.test_false_true_negative_positive(test_data, sig_prob, 0.988))
