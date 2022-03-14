import numpy as np
from numpy.polynomial import Legendre
import pandas as pd
from scipy.integrate import dblquad


bin_dic = {
    0: (0.1, 0.98),
    1: (1.1, 2.5),
    2: (2.5, 4),
    3: (4, 6),
    4: (6, 8),
    5: (15, 17),
    6: (17, 19),
    7: (11, 12.5),
    8: (1, 6),
    9: (15, 17.9)
}

def rescale_q2(q2_array):
    return (q2_array-9.55)/9.45

def make_Leg(poly_degree):
    P = []
    for i in range(poly_degree):
        Leg_int_coeff = np.zeros(i+1)
        Leg_int_coeff[-1] = 1
        P.append(Legendre(Leg_int_coeff))
    return P

def calc_coeff(dataframe, leg_shape=(6, 5)):
    '''
    Calculates coefficients for Legendre fitting.

    Parameters
    ----------
    dataframe : pandas DataFrame
        with columns of q2, costhetal, costhetak, phi
    leg_shape : order of Legendre polynomials to fit to

    Returns
    -------
    2D array, with shape given by `leg_shape`
    '''

    N = len(dataframe)
    q2 = rescale_q2(dataframe['q2'])
    ctl = dataframe['costhetal'].to_numpy()

    P = make_Leg(max(leg_shape)) # list of Legendre polynomials

    max_arange = np.arange(max(leg_shape))
    ij = [max_arange[:num]+1/2 for num in leg_shape] # '(2i+1)/2' factors
    ij_prod = np.einsum('i,j->ij', *ij, optimize=True)

    # outer product of 4 vectors in einsum notation
    # used einsum here because it's faster in this case

    P_i = np.array([P[i](q2) for i in range(leg_shape[0])])
    P_j = np.array([P[j](ctl) for j in range(leg_shape[1])])

    # equivalent einsum for this part – outer product and sum across data
    # P_ij_prod = np.einsum('ab,cb->ac', P_i, P_j, optimize=True)

    c = 1/N * ij_prod * (P_i[:,None]*P_j).sum(-1)

    return c

def acceptance_function(q2, ctl, coeff):
    '''
    Continuous acceptance function.

    Parameters
    ----------
    q2, ctl : int/float/1D array with same size
        values to evaluate the acceptance function at
    coeff : ndarray
        coefficients for fitted Legendre polynomials

    Returns
    -------
    1D array, with len of inputs
    '''

    # make sure all variables are numpy arrays
    q2 = np.asarray(q2)
    ctl = np.asarray(ctl)

    q2 = rescale_q2(q2)
    shape = coeff.shape

    P = make_Leg(max(shape))

    # # creating 8d arrays here!
    # P_i = np.array([P[i](q2) for i in range(shape[0])]).reshape(
    #     -1, 1, 1, 1, q2.size)
    # P_j = np.array([P[j](ctl) for j in range(shape[1])]).reshape(
    #     1, -1, 1, 1, ctl.size)
    # P_k = np.array([P[k](ctk) for k in range(shape[2])]).reshape(
    #     1, 1, -1, 1, ctk.size)
    # P_l = np.array([P[l](phi) for l in range(shape[3])]).reshape(
    #     1, 1, 1, -1, phi.size)

    # coeff = np.expand_dims(coeff, axis=4)
    # acc_func = (coeff * P_i*P_j*P_k*P_l).sum((0,1,2,3))

    P_i = np.array([P[i](q2) for i in range(shape[0])])
    P_j = np.array([P[j](ctl) for j in range(shape[1])])

    # again, code using einsum
    # convert 1d arrays to 2d, if not already 2d
    for P_var in (P_i, P_j):
        if P_var.ndim != 2:
            P_var.shape = (-1, 1)

    # sum across all coeffs, leaving q2, ctl, ctk and phi values
    acc_func = np.einsum(
        'ab,cb,ac->b', P_i, P_j, coeff, optimize='greedy')

    return acc_func

def q2_binned(df):
    '''
    Bin a given dataframe into q² bins.

    Parameters
    ----------
    df : dataframe/numpy array
        dataset to split into q² bins

    Returns
    -------
    dictionary
        with integer keys from 0..9, corresponding to the q² bins given in
        the TBPS website, also contains i1,i2,i3,i4, which are 'invalid' bins
    '''

    # bins ordered different, because np.histogram needs a
    # monotonically increasing array
    q2_bins_0 = [0.1, 0.98, 1.1, 2.5, 4, 6, 8, 15, 17, 19]
    q2_bins_1 = [1, 6, 11, 12.5, 15, 17.9]

    binned_0 = pd.cut(
        df['q2'], bins=q2_bins_0,
        labels=[0, 'i1', 1, 2, 3, 4, 'i2', 5, 6]
    )
    binned_1 = pd.cut(
        df['q2'], bins=q2_bins_1,
        labels=[8, 'i3', 7, 'i4', 9]
    )

    df_q2_bins_0 = dict(tuple(df.groupby(binned_0)))
    df_q2_bins_1 = dict(tuple(df.groupby(binned_1)))

    df_q2_binned = {**df_q2_bins_0, **df_q2_bins_1}

    return df_q2_binned

def decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9,
                 acceptance, q2, ctl, ctk, phi, coeff):
    '''
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    '''
    stl = np.sqrt(1 - ctl * ctl)
    stk = np.sqrt(1 - ctk * ctk)
    c2tl = 2 * ctl * ctl - 1
    s2tk = 2 * stk * ctk
    s2tl = 2 * stl * ctl
    stl_sq = stl * stl
    stk_sq = stk * stk
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    # equation from page 44
    scalar_array = 9 * np.pi / 32 * acceptance(q2, ctl, ctk, phi, coeff) * \
        (3/4 * (1 - F_l) * stk_sq +
        F_l * ctk * ctk +
        1/4 * (1 - F_l) * stk_sq * c2tl -
        F_l * ctk * ctk * c2tl +
        S_3 * stk_sq * stl_sq * np.cos(2 * phi) +
        S_4 * s2tk * s2tl * cphi +
        S_5 * s2tk * stl * cphi +
        4/3 * A_fb * stk_sq * ctl +
        S_7 * s2tk * stl * sphi +
        S_8 * s2tk * s2tl * sphi +
        S_9 * stk_sq * stl_sq * 2 * sphi * cphi)

    return scalar_array

def log_likelihood_S(df, coeff, F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9, _bin):
    '''
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    '''
    binnum = int(_bin)
    _bin = df[binnum]
    ctl = _bin['costhetal'].to_numpy()
    ctk = _bin['costhetak'].to_numpy()
    phi = _bin['phi'].to_numpy()
    q2 = _bin['q2'].to_numpy()

    print(F_l, end=', ')

    normalised_scalar_array = decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7,
    S_8, S_9, acceptance_function, q2, ctl, ctk, phi, coeff)

    def int_func(x):
        return decay_rate_S(F_l, A_fb, S_3, S_4, S_5, S_7, S_8, S_9,
        acceptance_function, *x, coeff)

    norm = vegas.Integrator(
        [ # integral limits for q2, ctl, ctk, phi
            [bin_dic[binnum][0], bin_dic[binnum][1]],
            [-1, 1],
            [-1, 1],
            [-np.pi, np.pi]
        ]
    )
    result = norm(int_func, nitn=10, neval=100)

    # print(result.summary())
    # print(f'result = {result}, Q = {result.Q:.2f}')

    normalised_scalar_array = normalised_scalar_array / result[0].mean

    return -np.log(normalised_scalar_array).sum(-1)

def decay_rate(fl, afb, q2, ctl, coeff):
    c2tl = 2 * ctl ** 2 - 1

    acceptance = acceptance_function(q2, ctl, coeff)

    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +
                   8/3 * afb * ctl) * acceptance
    return scalar_array

def d2gamma_p_d2q2_dcostheta(fl, afb, q2, ctl, coeff, _bin):
    scalar_array = decay_rate(fl, afb, q2, ctl, coeff)

    def integ(q2, ctl):
        return decay_rate(fl, afb, q2, ctl, coeff)

    # double integral over q² and ctl
    norm = dblquad(integ, bin_dic[_bin][0], bin_dic[_bin][1], -1, 1)[0]
    # norm = 1

    return scalar_array/norm

def log_likelihood(df, coeff, fl, afb, _bin):
    '''
    Returns the negative log-likelihood of the pdf defined above
    :param df: pandas dataFrame
    :param coeff: coefficients as ndarray of Legendre fits
    :param fl: f_l observable
    :param afb: a_fb observable
    '''

    _bin = int(_bin) # make sure index is an integer
    bin_data = df[_bin]

    # extract required data
    q2 = bin_data['q2'].to_numpy()
    ctl = bin_data['costhetal'].to_numpy()

    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(
        fl, afb, q2, ctl, coeff, _bin)

    NLL = -np.log(normalised_scalar_array, 
                  out=np.zeros((fl.size, afb.size, q2.size)),
                  where=(normalised_scalar_array>0)
            ).sum(-1)

    return NLL

if __name__ == '__main__':
    # for testing code

    df = pd.DataFrame()
    df_len = 34
    rng = np.random.default_rng(0)
    df['q2'] = rng.uniform(1, 16, df_len)
    df['costhetal'] = rng.uniform(-1, 1, df_len)
    # df['costhetak'] = rng.uniform(-1, 1, df_len)
    # df['phi'] = rng.uniform(-1, 1, df_len) * np.pi

    coeff = calc_coeff(df)
    # print(coeff[0])

    new = acceptance_function(df['q2'], df['costhetal'], coeff)

    print(new)
    # 0.48022672, 0.12913039, 0.20780453

