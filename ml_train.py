import ml_load
import ml_tools
from pyexpat import model
import xgboost
import json
import os
import numpy as np

"""
train and validate models using params from BayesianOptimization
"""

def ml_train_validate_to_be_optimized(**hyperparams):
    """
    Convert some hyperparams to integer values for ml_train_validate
    """
    hyperparams['n_estimators'] = int(hyperparams['n_estimators'] )
    hyperparams['max_depth'] = int(hyperparams['max_depth'])

    return ml_train_validate(**hyperparams)


def ml_train_validate(**hyperparams):

    # 1. get data
    train_data, validate_data, test_data = (
        ml_load.get_train_validate_test_for_all_peaking_bks(train_samples_limit=1000)
    )

    # 2. settings
    xgboost.set_config(verbosity=2)
    xge_model = xgboost.XGBClassifier(**hyperparams)

    # 3. train model
    print('Starting model training')
    # Some other model params are passed here
    ml_tools.ml_train_model(train_data, xge_model,
        eval_metric='logloss',
    )
    print('Completed model training')

    # 4. check with validate data
    sig_prob = ml_tools.ml_get_model_sig_prob(validate_data, xge_model)

    # best sb model can give
    # Jose
    threshold_list = np.linspace(0.8,1,600)
    sb_list = []
    for thresh in threshold_list:
        sb_list.append(ml_tools.test_sb(validate_data, sig_prob, thresh))
    # Finding best value of threshold to optimise SB metric
    bestIx = np.nanargmax(np.array(sb_list))
    bestSb = sb_list[bestIx]
    print('sb', bestSb)

    # 5. save model
    MODEL_FILE_NAME = 'peaking_sb_{}_'.format(bestSb)
    for i in hyperparams:
        MODEL_FILE_NAME = MODEL_FILE_NAME + str(i) +'_'+ str(hyperparams[i]) +'_'
    MODEL_FILE_NAME = MODEL_FILE_NAME + '.model'

    xge_model.save_model(os.path.join('optimisation_models',MODEL_FILE_NAME))

    return bestSb


# examples
# bounded region of hyperparameters - arbitrary
pbounds = {
    'n_estimators':(100,200),
    'subsample':(0.5,1),
    'max_depth':(6,20),
    'learning_rate':(0.01, 0.3),
    # 'gamma':(0,0.02),
    # 'reg_alpha':(0,3),
    # 'reg_lambda':(1,4),
    }

# bayesian_nextpoint -- example 1
# util_args = {'kind':"ucb", 'kappa':2.5,'xi':0.0}
# next_point = ml_tools.bayesian_nextpoint(ml_train_validate_to_be_optimized, pbounds, random_state=None, **util_args)
# ml_train_validate_to_be_optimized(**next_point)  # train model

# bayesian_optimisation -- example 2
# ml_tools.bayesian_optimisation(ml_train_validate_to_be_optimized, pbounds)
