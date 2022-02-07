
from ml_recreate import fit_new_model, load_train_validate_test, concat_datasets, combine_signal_background, optimize_threshold, predict_prob
from core import RAWFILES
import ml_tools, ml_combinatorial_extraction
import os

"""
train and validate models using params from BayesianOptimization
"""

TRAIN_COMB_BK = False

SAVE_FOLDER = 'optimisation_models'
if TRAIN_COMB_BK:
    SAVE_FOLDER = 'optimisation_models_comb'

def ml_train_validate_combinatorial(**hyperparams):
    train, validate, test = (
        ml_combinatorial_extraction.load_combinatorial_train_validate_test(train_samples_limit=None))

    return ml_train_validate(train, validate, **hyperparams)

def ml_train_validate_peaking(**hyperparams):
    # 1. get data
    signal = load_train_validate_test(RAWFILES.SIGNAL)
    background = concat_datasets([load_train_validate_test(file) for file in RAWFILES.peaking_bks])
    train, validate, test = combine_signal_background(signal, background)

    return ml_train_validate(train, validate, **hyperparams)


def ml_train_validate(train_data, validate_data, **hyperparams, ):
    # Convert some hyperparams to integer values
    hyperparams['n_estimators'] = int(hyperparams['n_estimators'] )
    hyperparams['max_depth'] = int(hyperparams['max_depth'])

    # 3. train model
    xgb_model = fit_new_model(train_data, **hyperparams)

    # 4. check with validate data
    sig_prob = predict_prob(validate_data, xgb_model)
    thresh = optimize_threshold(validate_data, sig_prob)
    
    # best sb model can give
    # Jose
    bestSb = ml_tools.test_sb(validate_data, sig_prob, thresh)
    print('Model SB:', bestSb)
    print('Best thresh:', thresh)
    print('Model stats:', ml_tools.test_false_true_negative_positive(validate_data, sig_prob, thresh))

    # 5. save model
    MODEL_FILE_NAME = 'peaking_sb_{}_'.format(bestSb)
    if TRAIN_COMB_BK:
        MODEL_FILE_NAME = 'comb_sb_{}_'.format(bestSb)
    for i in hyperparams:
        MODEL_FILE_NAME = MODEL_FILE_NAME + str(i) +'_'+ str(hyperparams[i]) +'_'
    MODEL_FILE_NAME = MODEL_FILE_NAME + '.model'

    xgb_model.save_model(os.path.join(SAVE_FOLDER,MODEL_FILE_NAME))

    return bestSb


if __name__ == '__main__':
    # bounded region of hyperparameters - arbitrary
    pbounds = {
        'n_estimators':(100,500),
        'max_depth':(3,8),
        'learning_rate':(0.01, 0.3),
        'gamma':(0,0.02),
        'reg_alpha':(0,3),
        'reg_lambda':(1,4),
        }

    # bayesian_optimisation
    if TRAIN_COMB_BK:
        train_func = ml_train_validate_combinatorial
    else:
        train_func = ml_train_validate_peaking

    optimizer = ml_tools.bayesian_optimisation(
        train_func,
        pbounds,
        log_folder = os.path.join(SAVE_FOLDER),
        bool_load_logs = True,
        explore_runs = 5, exploit_runs = 20
        )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print('Best performance:', optimizer.max)