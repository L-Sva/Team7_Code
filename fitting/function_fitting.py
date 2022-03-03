import numpy as np
from scipy.integrate import quad

from functions import acceptance_function, q2bins, rescale_q2

def raw_d2(ctl, fl, afb, _bin, q_norm, params_dict):
    c2tl = 2 * ctl ** 2 - 1
    acceptance = acceptance_function(q_norm[_bin], ctl, params_dict)
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl) * acceptance
    return scalar_array

def d2gamma_p_d2q2_dcostheta(fl, afb, ctl, _bin, q_norm, params_dict):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    """
    # normalise array to account for the non-unity acceptance function
    normalised_scalar_array = raw_d2(
        ctl, fl, afb, _bin, q_norm, params_dict).squeeze() / \
    quad(raw_d2, -1, 1, args = (fl, afb, _bin, q_norm, params_dict))[0]

    return normalised_scalar_array

def log_likelihood(df, params_dict, fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param df: q² binned dataset
    :param q_norm: rescaled (between [-1, +1]) q² values to
    evaluate acceptance function at
    :param params_dict: dictionary with required values for calculating the
    acceptance function. Needs keys: P and c, automatically added when
    find_acceptance.py is run.
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    """
    
    q2_bins_mid = (q2bins[:,:-1]+q2bins[:,1:])/2
    q_norm = rescale_q2(q2_bins_mid).flatten()
        
    _bin = int(_bin) # make sure index is an integer

    bin_data = df[str(_bin)]
    ctl = bin_data['costhetal'].to_numpy()
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(
        fl=fl, afb=afb, ctl=ctl, _bin=_bin,
        q_norm=q_norm, params_dict=params_dict
    )
    with np.errstate(invalid='ignore'): # ignore 'invalid log' message
        NLL = -np.log(normalised_scalar_array).sum(-1)
    return NLL