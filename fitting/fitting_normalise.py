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

def d2gamma(ctl, q2, fl, afb):
    '''
    returns:
            d2gamma (not normalised)
    '''
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl)

    return scalar_array

def d2gamma_withAcceptance(ctl, q2, fl, afb):
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
    returns:
            d2gamma* acceptance (normalised)
    '''
    # step0: set integral limits
    # cos_theta_l - integral boundary: -1 to 1
    ctl_lower = -1 # change this afterwards if related to "ctl" value of each bin
    ctl_higher = 1

    # q2 - integral limits
    q2bins = np.array(
        [[0.1, 0.98],
         [1.1, 2.5],
         [2.5, 4.0],
         [4.0, 6.0],
         [6.0, 8.0],
         [15.0, 17.0],
         [17.0, 19.0],
         [11.0, 12.5],
         [1.0, 6.0],
         [15.0, 17.9]]
    )


    # step1: integrate d2gamma
    gamma = 0
    for i in range(0, 10):  # iterate over all bins
        dummy, dummy_err =  integrate.dblquad(d2gamma, q2bins[i][0], q2bins[i][1], ctl_lower, ctl_higher, args=(fl, afb))
        gamma += dummy
    print(gamma)

    # step2: integrate d2gamma * acceptance
    gamma_withAcceptance = 0
    for i in range(0,10):
        dummy, dummy_err = integrate.dblquad(d2gamma_withAcceptance,  q2bins[0][0], q2bins[0][1], ctl_lower, ctl_higher, args=(fl, afb))
        gamma_withAcceptance += dummy
    # print(gamma_withAcceptance)

    # step3: normalisation

    # step4: return "normalised d2gamma_withAcceptance" value with ctl, fl, afb given
    return





# ----------------- tests ------------------
# _test_bin = 1
# _test_ctl = ctl_finder(_test_bin)  # question: why 100+ ctl values for each bin??

_test_ctl = [-0.33578466]* 10
_test_afb = 0.7
_test_fl = 0.0

d2gamma_withAcceptance_normalised(_test_ctl, _test_fl, _test_afb)
