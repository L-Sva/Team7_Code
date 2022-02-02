
from core import RAWFILES, load_file
from histrogram_plots import generic_selector_plot
import ml_tools
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import os
import ml_load, ml_combinatorial_extraction

# Use module to get prepared train/test data
train_data, validate, test_data = (
    ml_combinatorial_extraction.load_combinatorial_train_validate_test(
        train_samples_limit=None, remove_columns=True)
)
_, _, all_col_test_data = (
    ml_combinatorial_extraction.load_combinatorial_train_validate_test(
        train_samples_limit=None, remove_columns=False)
)
# Make more logging information available
xgboost.set_config(verbosity=2)

# Initialise model, some model parameters are passed here
xge_model = xgboost.XGBClassifier(
    max_depth=10,
)

LOAD_FROM_SAVED = True
MODEL_FILE_NAME = 'comb_hyperparameters_opt_best.model'

if LOAD_FROM_SAVED:
    xge_model.load_model(os.path.join('examples_save',MODEL_FILE_NAME))

else:
    print('Starting model training')
    # Some other model params are passed here
    ml_tools.ml_train_model(train_data, xge_model, 
        eval_metric='logloss',
    )
    print('Completed model training')

    xge_model.save_model(os.path.join('examples_save',MODEL_FILE_NAME))

# Model performance analysis

sig_prob = ml_tools.ml_get_model_sig_prob(test_data, xge_model) # signal probability from bdt model

# plotting roc_curve
roc_curve_res = ml_tools.roc_curve(xge_model, test_data)
ml_tools.plot_roc_curve(roc_curve_res['fpr'],roc_curve_res['tpr'],roc_curve_res['area'])
plt.show()

# plot of sb vs threshold
# TODO: Convert to general use function and move into ml_tools
threshold_list = np.linspace(0.1,1,600)
sb_list = []
for thresh in threshold_list:
    sb_list.append(ml_tools.test_sb(test_data, sig_prob, thresh))

plt.plot(threshold_list, sb_list)
plt.xlabel('Cut value')
plt.ylabel('$S / \\sqrt{S + B}$')
plt.show()
plt.close()

# Finding best value of threshold to optimise SB metric
# TODO: Make function in ml_tools
bestIx = np.nanargmax(np.array(sb_list))
bestCut = threshold_list[bestIx]

# Printout
print("ML selector only")
print(ml_tools.test_false_true_negative_positive(test_data, sig_prob, bestCut))
print('SB quality metric',ml_tools.test_sb(test_data, sig_prob, bestCut))

# Labeling the features from pandas dataframe
xge_model.get_booster().feature_names = [x for x in train_data.drop('category', axis=1)]

total = load_file(RAWFILES.TOTAL_DATASET)
total_red = ml_tools.ml_strip_columns(total, reject_column_names=('B0_MM', 'Kstar_MM'))

# Values cutting
sig_prob = ml_tools.ml_get_model_sig_prob(total_red, xge_model)
generic_selector_plot(
    total, 
    total[sig_prob > bestCut], 
    total[sig_prob < bestCut],
    'B0_MM')
plt.show()

# Plotting the 'importance' of each feature
# TODO: How is importance determined? (low priority)
xgboost.plot_importance(xge_model, max_num_features=20)
plt.tight_layout()
plt.show()