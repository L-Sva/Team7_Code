import numpy as np
import matplotlib.pyplot as plt
import pickle
from iminuit import Minuit
from scipy.integrate import quad
from functools import partial

from Team7_Code.core import load_file, RAWFILES
from Team7_Code.ES_functions.Compiled import selection_all
from functions import q2_binned, acceptance_function, q2bins, rescale_q2
from function_fitting import log_likelihood

plt.rcParams['font.size'] = 18

if __name__=='__main__':
    with open('tmp/acceptance_coeff.pkl', 'rb') as f:
        params_dict = pickle.load(f)

    # q² bins to evaluate the acceptance function at
    q2_bins_mid = (q2bins[:,:-1]+q2bins[:,1:])/2
    q0_norm = rescale_q2(q2_bins_mid).flatten()

    raw_total = load_file(RAWFILES.TOTAL_DATASET)
    filtered_total, _ = selection_all(raw_total)

    q2_filtered = q2_binned(filtered_total)

    # so that I don't have to change all the code below
    df_log_likelihood = partial(
        log_likelihood, q2_filtered, q0_norm, params_dict)

    _test_bin = 1

    #following values are used just for fixing one parameter while the other is varied
    _test_afb = 0.7
    _test_fl = 0.0

    x = np.linspace(-1, 1, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #testing log_likelihood for different values of fl from -1 to 1
    ax1.plot(x, [df_log_likelihood(fl=i, afb=_test_afb, _bin=_test_bin) for i in x])
    ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
    ax1.set_xlabel(r'$F_L$')
    ax1.set_ylabel(r'$-\mathcal{L}$')
    ax1.grid()

    #testing log_likelihood for different values of afb from -1 to 1
    ax2.plot(x, [df_log_likelihood(fl=_test_fl, afb=i, _bin=_test_bin) for i in x])
    ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
    ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
    ax2.set_xlabel(r'$A_{FB}$')
    ax2.set_ylabel(r'$-\mathcal{L}$')
    ax2.grid()
    plt.tight_layout()
    plt.show()


    #these 2 are only used for checking in the next cell
    bin_number_to_check = 0  # bin that we want to check in more details in the next cell
    bin_results_to_check = None


    df_log_likelihood.errordef = Minuit.LIKELIHOOD #set to a negative log likelihood function
    decimal_places = 3 #only used for outputting results
    starting_point = [-0.1,0.0] #starting for fl and afb respectively

    #setting up lists for outputting of results
    fls, fl_errs = [], []
    afbs, afb_errs = [], []

    for i in range(10):
        m = Minuit(df_log_likelihood, fl=starting_point[0], afb=starting_point[1], _bin=i)
        m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
        m.limits=((-1.0, 1.0), (-1.0, 1.0), None) #setting the limits for parameters fl, afb, and bin number in that order
        m.migrad() #find min using gradient descent
        m.hesse() #finds estimation of errors at min point by looking at th
        if i == bin_number_to_check:
            bin_results_to_check = m #sets up results for checking in the next cell

        fls.append(m.values[0])
        afbs.append(m.values[1])
        fl_errs.append(m.errors[0])
        afb_errs.append(m.errors[1])

        print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")

    fits = {'fls': fls, 'fl_errs': fl_errs, 'afbs': afbs, 'afb_errs': afb_errs}

    with open('tmp/af_fitting.pkl', 'wb') as f:
        pickle.dump(fits, f)

    # actual (given values), used for comparison
    FL_aval = [
        0.2964476598667644, 0.7603956395622157, 0.7962654932624427,
        0.7112903962261427, 0.6069651454293227, 0.3484407559729121,
        0.32808100221720543, 0.4351903815658659, 0.7476437141490421,
        0.34015604925198045
    ]
    FL_aerr = [
        0.05064238707746488, 0.04317410278778188, 0.03363967369234135,
        0.04909550894620607, 0.050785046867782326, 0.03554031324502086,
        0.02798976439727086, 0.03663667651413794, 0.03988961460715321,
        0.024825565828724583
    ]
    AFB_aval = [
        -0.0970515604684916, -0.13798671499985016, -0.017385033593575933,
        0.12215506287920677, 0.23993949201268056, 0.4019144709381388,
        0.3183907920796311, 0.3913899537158143, 0.0049291176692496394,
        0.3676716852255129
    ]
    AFB_aerr = [
        0.008421043319078612, 0.03195845613292291, 0.029395083110202395,
        0.03961889843177776, 0.04728136705066239, 0.030140936643680404,
        0.03393144831695434, 0.023627117754196975, 0.02805838529981395,
        0.03223029184299107
    ]


    with open('tmp/af_fitting.pkl', 'rb') as f:
        fits = pickle.load(f)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].errorbar(
        range(10), fits['fls'], yerr=fits['fl_errs'], fmt='k.', ms=5, capsize=3)
    ax[0].errorbar(range(10), FL_aval, yerr=FL_aerr, fmt='r.', ms=5, capsize=3)
    ax[0].set(xlabel='Bin number', ylabel=r'$F_L$', xticks=range(10))
    ax[0].grid()
    # ax[0].plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    ax[1].errorbar(
        range(10), fits['afbs'], yerr=fits['afb_errs'], fmt='k.', ms=5, capsize=3)
    ax[1].errorbar(range(10), AFB_aval, yerr=AFB_aerr, fmt='r.', ms=5, capsize=3)
    ax[1].grid()
    ax[1].set(xlabel='Bin number', ylabel=r'$A_{FB}$', xticks=range(10))

    fig.suptitle('Our fit (black), SM Prediction (red)', size=17)

    plt.show()

