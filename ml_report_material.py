import matplotlib.pyplot as plt
from core import RAWFILES, ensure_dir
from ml_main import (
    load_train_validate_test, concat_datasets, combine_signal_background, predict_prob, plot_roc_curve
)
import ml_combinatorial_extraction
from ml_tools import (
    ideal_sb, test_sb
)
from ml_selector import (
    pk_bk_model, comb_bk_model, PK_THRESH, COMB_THRESH,
)
from os.path import join

ROC_CURVE_DIR = 'roc_curves'

# Peaking background selector
ensure_dir(ROC_CURVE_DIR)

signal = load_train_validate_test(RAWFILES.SIGNAL)
background = concat_datasets([load_train_validate_test(file) for file in RAWFILES.peaking_bks])
train, validate, test = combine_signal_background(signal, background)
prob = predict_prob(test, pk_bk_model)
print(
    'Peaking background SB metric:\n',
    test_sb(test, prob, PK_THRESH)
)
print(
    'Ideal SB:\n',
    ideal_sb(test)
)
plot_roc_curve(pk_bk_model, test)
plt.savefig(join(ROC_CURVE_DIR,'peaking_background.png'))
plt.show()

# Combinatorial background selector
train, validate, test = (
    ml_combinatorial_extraction.load_combinatorial_train_validate_test(train_samples_limit=None))
prob = predict_prob(test, comb_bk_model)
print(
    'Combinatorial background SB metric:\n',
    test_sb(test, prob, COMB_THRESH)
)
print(
    'Ideal SB:\n',
    ideal_sb(test)
)
plot_roc_curve(comb_bk_model, test)
plt.savefig(join(ROC_CURVE_DIR,'comb_background.png'))
plt.show()