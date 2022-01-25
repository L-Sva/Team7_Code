from histrogram_plots import generic_selector_plot
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

LOAD = True
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

roc_curve_res = ml_tools.roc_curve(test_data, sig_prob)
ml_tools.plot_roc_curve(roc_curve_res['fpr'],roc_curve_res['tpr'],roc_curve_res['area'])
plt.show()

threshold_list = np.linspace(0.8,1,600)
sb_list = []
for thresh in threshold_list:
    sb_list.append(ml_tools.test_sb(test_data, sig_prob, thresh))

plt.plot(threshold_list, sb_list)
plt.xlabel('Cut value')
plt.ylabel('$S / \\sqrt{S + B}$')
plt.show()
plt.close()

bestIx = np.nanargmax(np.array(sb_list))
bestCut = threshold_list[bestIx]

print("ML selector only")
print(ml_tools.test_false_true_negative_positive(test_data, sig_prob, bestCut))
print('SB quality metric',ml_tools.test_sb(test_data, sig_prob, bestCut))

xge_model.get_booster().feature_names = [x for x in train_data.drop('category', axis=1)]

generic_selector_plot(test_data, test_data[sig_prob > bestCut], test_data[sig_prob < bestCut],'q2')
plt.show()

# xgboost.plot_tree(xge_model, rankdir='LR')
# plt.show()

xgboost.plot_importance(xge_model, max_num_features=20)
plt.tight_layout()
plt.show()

print('Combining ML and q2 selectors')
q2 = test_data["q2"]
sig_prob[np.bitwise_and(q2 > 8, q2 < 11)] = 0
sig_prob[np.bitwise_and(q2 > 12.5, q2 < 15)] = 0
print(ml_tools.test_false_true_negative_positive(test_data, sig_prob, bestCut))
print('SB quality metric',ml_tools.test_sb(test_data, sig_prob, bestCut))