import numpy as np
import pickle
import scipy.integrate as integrate

from core import load_file, RAWFILES
from ES_functions.Compiled import selection_all
from functions import q2_binned, acceptance_function, q2bins, rescale_q2





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

    acceptance = acceptance_function(q0_norm[0], ctl, params_dict)


    # find "d2gamma * acceptance"
    c2tl = 2 * ctl ** 2 - 1

    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl) * acceptance

    return scalar_array

def d2gamma_withAcceptance_normalised(ctl, fl, afb):
    '''
    inputs:
        ctl - array
        fl - float
        afb - float

        (for a single bin)

    returns:
            normalisation constant, and
            values of d2gamma* acceptance (normalised)

            （for a list of ctl values,
            for a single bin）
    '''

    # step0: set integral limits
    ctl_lower = -1
    ctl_higher = 1

    # step1: integrate d2gamma (wrt ctl)
    # dummy, dummy_err = integrate.quad(d2gamma, ctl_lower, ctl_higher, args = (fl, afb))
    # gamma = dummy
    # print(gamma) # result = [1]*10 as expected


    # step2: integrate d2gamma * acceptance (wrt ctl)
    dummy, dummy_err = integrate.quad(d2gamma_withAcceptance, ctl_lower, ctl_higher, args = (fl, afb))
    gamma_withAcceptance = dummy
    # print('value:',gamma_withAcceptance)


    # step3: normalisation of d2gamma_withAcceptance
    result = []
    norm_const = 1.0/gamma_withAcceptance

    for i in range(len(ctl)):
        result.append(d2gamma_withAcceptance(ctl[i], fl, afb)[0][0] * norm_const)

    return norm_const, result





# ----------------- tests ------------------
_test_bin = 1
_test_ctl = ctl_finder(_test_bin)

# _test_ctl = [-0.33578466]* 10
_test_afb = -0.0970515604684916
_test_fl = 0.2964476598667644

norm_const, result = d2gamma_withAcceptance_normalised(_test_ctl, _test_fl, _test_afb)
print(result)
