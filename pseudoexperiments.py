import os
from pathlib import Path
import pickle
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import numpy as np
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
import lmfit

from core import RAWFILES, ensure_dir, load_file
from ES_functions.modifiedselectioncuts import selection_all, selection_all_withoutres, q2_resonances, selection_pb
from Fitting_New.functions_new import acceptance_function, calc_coeff, calc_coeff_4d, log_likelihood, q2_binned
from Fitting_New.integrals import get_reduced
from ml_selector import remove_combinatorial_background, remove_peaking_background
from histrogram_plots import plot_hist_quantity

@dataclass
class Fl_afb_fit():
    fl_best: List = None
    fl_best_err: List = None
    afb_best: List = None
    afb_best_err: List = None

    # Values of FL and AFB under standard model predicitons
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

    def plot_fits_by_bin(self):
        

        fig, ax = plt.subplots(1, 2, constrained_layout=True)

        ax[0].errorbar(
            range(10), self.fl_best, yerr=self.fl_best_err, fmt='k.', ms=5, capsize=3)
        ax[0].errorbar(range(10), self.FL_aval, yerr=self.FL_aerr, fmt='r.', ms=5, capsize=3)
        ax[0].set(xlabel='Bin number', ylabel=r'$F_L$', xticks=range(10))
        ax[0].grid()

        ax[1].errorbar(
            range(10), self.afb_best, yerr=self.afb_best_err, fmt='k.', ms=5, capsize=3)
        ax[1].errorbar(range(10), self.AFB_aval, yerr=self.AFB_aerr, fmt='r.', ms=5, capsize=3)
        ax[1].grid()
        ax[1].set(xlabel='Bin number', ylabel=r'$A_{FB}$', xticks=range(10))

        fig.suptitle('Our fit (black), SM Prediction (red)', size=17)

        plt.show()


class Experiment():
    def __init__(self, data: pd.DataFrame, acceptance_data: pd.DataFrame):

        # Only need to keep the angular quantity data
        angular_quantites = ['q2','costhetal','costhetak','phi']
        if data is not None:
            self.data = data[angular_quantites]
        self.acceptance_data = acceptance_data[angular_quantites]

        self.c = None
        self.delta_norm = None
        self.fl_afb_fit = None
        self.pseudoexperiments: list[Experiment] = []
        self.fit_psudoexp_acc: list[Fl_afb_fit] = []
        self.checkpoint: int = 0

    def fit_acceptance(self, leg_shape: Tuple[int, int] = (5, 6)):
        self.c: np.ndarray = calc_coeff(
            self.acceptance_data, leg_shape=leg_shape)
        self.delta_norm = get_reduced(self.c)
    
    def fit_afb_and_fl(self, acceptance_coeffs=None, delta_normed=None):
        if acceptance_coeffs is None:
            acceptance_coeffs = self.c
            delta_normed = self.delta_norm

        df_log_likelihood = partial(
            log_likelihood, q2_binned(
                self.data), acceptance_coeffs, delta_normed 
        )

        fls, fl_errs = [], []
        afbs, afb_errs = [], []

        for bin_number in range(10):
            m = Minuit(
                df_log_likelihood, 
                fl= 0, 
                afb= 0, 
                _bin=bin_number
            )
            # fixing the bin number as we don't want to optimize it
            m.fixed['_bin'] = True
            m.limits = ((-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad()  # find min using gradient descent
            m.hesse()  # finds estimation of errors at min point

            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])

        return Fl_afb_fit(fls, fl_errs, afbs, afb_errs)

    def fit_afb_and_fl_own_acceptance(self):
        if self.c is None:
            self.fit_acceptance()
        self.fl_afb_fit = self.fit_afb_and_fl(self.c, self.delta_norm)

    def make_and_fit_pseudoexperiments_acceptance(self, n = 100, leg_shape=(5,6)):
        self.pseudoexperiments = []
        
        for i in range(n):
            print(" Fitting acceptance of pseudoexperiments: ", i, end='\r')
            exp = Experiment(
                None, self._compute_pseudo_dataset_acceptance()
            )
            exp.fit_acceptance(leg_shape)
            del exp.acceptance_data
            self.pseudoexperiments.append(exp)

    def _compute_pseudo_dataset_acceptance(self):
        return self.acceptance_data.sample(
            frac=1, replace=True, axis=0)

        # acc = load_file(RAWFILES.ACCEPTANCE)
        # acc = acc.sample(frac=1, replace=True, axis=0)
        # s, ns = selection_all_withoutres(acc)
        # s_acc, ns = remove_combinatorial_background(s)
        # return s_acc

    def fit_afb_fl_pseduo_datasets_acceptance(self):
        fits = []

        for i, exp in enumerate(self.pseudoexperiments):
            print(" Fitting afb_fl for variying acceptance function: ", i, end='\r')
            fit = self.fit_afb_and_fl(exp.c, exp.delta_norm)
            fits.append(fit)

        self.fl_afb_fit_psudoexp_acc: list[Fl_afb_fit] = fits

def plot_ll_samples(exp: Experiment, _bin):
    afb = np.linspace(-1,1,10)
    fl = np.linspace(-1,1,10)
    AFB, FL = np.meshgrid(afb, fl)
    LL = [log_likelihood(q2_binned(exp.data), exp.c, exp.delta_norm, fl, afb, _bin) for fl,afb in zip(AFB.flatten(), FL.flatten())]
    LL = np.array(LL).reshape(AFB.shape)

    plt.pcolormesh(AFB, FL, LL, shading='nearest')
    cbar = plt.colorbar()
    plt.show()


if __name__ == '__main__':


    DIR = 'pseudoexperiments'
    FILE = Path(DIR, 'experiment-manualselector.pkl')
    acceptance_leg_shape = (5, 6)
    # FILE = Path(DIR, 'experiment-manualselector-altlegshape.pkl')
    # acceptance_leg_shape = (5+3, 6+3)
    # FILE = Path(DIR, 'experiment-manualselector-acc-full-recalc.pkl')
    # acceptance_leg_shape = (5, 6)
    ensure_dir(DIR)

    def dump_exp(file, exp: Experiment, checkpoint: int=0):
        exp.checkpoint = checkpoint
        with open(file, 'wb') as fileio:
            pickle.dump(exp,fileio)
    
    def load_exp(file):
        with open(file, 'rb') as fileio:
            return pickle.load(fileio)

    if not os.path.exists(FILE):
        total = load_file(RAWFILES.TOTAL_DATASET)
        s, ns = selection_all(total)
        s_total, ns = remove_combinatorial_background(s)

        acc = load_file(RAWFILES.ACCEPTANCE)
        s, ns = selection_all_withoutres(acc)
        s_acc, ns = remove_combinatorial_background(s)

        exp = Experiment(s_total, s_acc)

        dump_exp(FILE, exp, 0)
    else:
        exp =  load_exp(FILE)

    if exp.checkpoint < 1:
        exp.fit_acceptance()
        exp.fit_afb_and_fl_own_acceptance()
        dump_exp(FILE, exp, 1)

    if exp.checkpoint < 2: 
        exp.make_and_fit_pseudoexperiments_acceptance(5, acceptance_leg_shape)
        dump_exp(FILE, exp, 2)

    if exp.checkpoint < 3:
        exp.fit_afb_fl_pseduo_datasets_acceptance()
        dump_exp(FILE, exp, 3)

    v = [x.fl_best[9] for x in exp.fl_afb_fit_psudoexp_acc]
    n, binedges = np.histogram(v, bins=40)
    bincenters = (binedges[1:] + binedges[:-1])/2
    bincenters = bincenters * 1e9
    model = lmfit.models.GaussianModel()
    paras = model.guess(n, x=bincenters)
    res = model.fit(n, paras, x=bincenters)
    v = res.best_values
    v['center'] = v['center'] / 1e9
    v['sigma'] = v['sigma'] / 1e9
    print(v)

    res.plot()
    plt.show()

    exp.fl_afb_fit.plot_fits_by_bin()
