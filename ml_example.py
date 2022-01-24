import ml_tools
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import os
import ml_peakingbks_test_train

train_data, test_data = ml_peakingbks_test_train.get_test_train(train_samples_limit=100000)

xgboost.set_config(verbosity=2)
xge_model = xgboost.XGBClassifier(
    max_depth=10
)

LOAD = False
MODEL_NAME = '0004_peaking.model'
if LOAD:
    xge_model.load_model(os.path.join('examples_save',MODEL_NAME))

else:
    print('Starting model training')
    ml_tools.ml_train_model(train_data, xge_model, 
        eval_metric='logloss',
    )
    print('Completed model training')

    xge_model.save_model(os.path.join('examples_save',MODEL_NAME))

sig_prob = ml_tools.ml_get_model_sig_prob(test_data, xge_model) # signal probability from bdt model

threshold_list = np.linspace(0.8,1,600)
sb_list = []
for thresh in threshold_list:
    sb_list.append(ml_tools.test_sb(test_data, sig_prob, thresh))

plt.plot(threshold_list, sb_list)
plt.xlabel('Cut value')
plt.ylabel('$S / \\sqrt{S + B}$')
plt.show()

bestIx = np.nanargmax(np.array(sb_list))
bestCut = threshold_list[bestIx]

print(ml_tools.test_false_true_negative_positive(test_data, sig_prob, bestCut))
