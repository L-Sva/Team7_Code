import numpy as np
import pickle
import scipy.integrate as integrate

from core import load_file, RAWFILES
from ES_functions.Compiled import selection_all
from functions import q2_binned, acceptance_function, q2bins, rescale_q2



# ==================================================
# question: why q0_norm[0]?? the first element???
# question: why 100+ ctl values for each bin??
# ==================================================


'''
def ctl_finder( _bin ):

    # extracted from .../fitting/main.py

    # 1. find q2_filtered
    raw_total = load_file(RAWFILES.TOTAL_DATASET)  # raw data
    filtered_total, _ = selection_all(raw_total)   # signal data - from raw data using selection cuts
    q2_filtered = q2_binned(filtered_total)

    # 2. find ctl
    df = q2_filtered
    _bin = df[str(int(_bin))]
    ctl = _bin['costhetal'].to_numpy()

    return ctl
'''


def d2gamma(ctl, fl, afb):
    '''
    returns:
            d2gamma (not normalised)
    '''
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl)

    return scalar_array

def d2gamma_withAcceptance(ctl, fl, afb):
    """
    returns:
            d2gamma * acceptance function (not normalised)
    """
    # find "acceptance"
    q2_bins_mid = (q2bins[:,:-1]+q2bins[:,1:])/2    # extracted from .../fitting/main.py
    q0_norm = rescale_q2(q2_bins_mid).flatten()

    with open('tmp/acceptance_coeff.pkl', 'rb') as f:
        params_dict = pickle.load(f)

    acceptance = acceptance_function(q0_norm[0], ctl, params_dict) # question: why q0_norm[0]?? the first element???


    # find "d2gamma * acceptance"
    c2tl = 2 * ctl ** 2 - 1

    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl) * acceptance

    return scalar_array

def d2gamma_withAcceptance_normalised(ctl, fl, afb):
    '''
    inputs:
        ctl - array, len = 10
        fl - array, len = 10
        afb - arry, len = 10
    returns:
            d2gamma* acceptance (normalised)
    '''

    # step0: set integral limits
    ctl_lower = -1
    ctl_higher = 1

    # step1: integrate d2gamma (wrt ctl)
    gamma = []
    for i in range(0,10):
        dummy, dummy_err = integrate.quad(d2gamma, ctl_lower, ctl_higher, args = (fl[i], afb[i]))
        gamma.append(dummy)
    # print(gamma) # result = [1]*10 as expected


    # step2: integrate d2gamma * acceptance (wrt ctl)
    gamma_withAcceptance = []
    for i in range(0,10):
        dummy, dummy_err = integrate.quad(d2gamma_withAcceptance, ctl_lower, ctl_higher, args = (fl[i], afb[i]))
        gamma_withAcceptance.append(dummy)
    # print(gamma_withAcceptance)

    # step3: normalisation of d2gamma_withAcceptance
    result = []
    factor = 0
    for i in range(0, 10):
        factor = 1.0/gamma_withAcceptance[i]  #normalisation factor for each bin
        # print(gamma_withAcceptance[i] * factor) # check if normalised to 1
        result.append(d2gamma_withAcceptance(ctl[i], fl[i], afb[i])[0][0] * factor)

    return result





# ----------------- tests ------------------
# _test_bin = 1
# _test_ctl = ctl_finder(_test_bin)  # question: why 100+ ctl values for each bin??

_test_ctl = [-0.33578466]* 10
_test_afb = [-0.0970515604684916, -0.13798671499985016, -0.017385033593575933,
0.12215506287920677, 0.23993949201268056, 0.4019144709381388,
0.3183907920796311, 0.3913899537158143, 0.0049291176692496394,
0.3676716852255129]
_test_fl = [0.2964476598667644, 0.7603956395622157, 0.7962654932624427,
0.7112903962261427, 0.6069651454293227, 0.3484407559729121,
0.32808100221720543, 0.4351903815658659, 0.7476437141490421,
0.34015604925198045]

d2gamma_withAcceptance_normalised(_test_ctl, _test_fl, _test_afb)
