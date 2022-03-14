from dataclasses import dataclass
from functools import partial
import os
import pickle
from typing import List
from ES_functions.Compiled import q2_resonances, selection_all
from core import RAWFILES, ensure_dir, load_file
from fitting.function_fitting import log_likelihood
from ml_selector import remove_all_bk
from iminuit import Minuit

def compute_pseudoexperiments(data,n=50):
    pseudoexperiments = [data.sample(frac=1, replace=True, axis=0) for i in range(n)]
    return pseudoexperiments

def cached_compute_pseudoexperiments(data,cache_file_path,n=50):
    if not os.path.exists(cache_file_path):
        pseudoexperiments = compute_pseudoexperiments(data,n)
        with open(cache_file_path, 'wb') as file:
            pickle.dump(pseudoexperiments, file)
    else:
        with open(cache_file_path, 'wb') as file:
            pseudoexperiments = pickle.load(file)

@dataclass
class fl_afb_fit():
    fl_best: List = None
    fl_best_err: List = None
    afb_best: List = None
    afb_best_err: List = None
    likelihood_best: List = None

class Experiment():
    def __init__(self,data):
        self.data_with_resonances = data
        self.data_without_resonasces, _ = q2_resonances(self.data_with_resonances)
        
    def set_acceptance_params(self,params):
        self.acceptance_params = params
    
    def fit_ab_and_fl(self, acceptance_params):

        df_log_likelihood = partial(
            log_likelihood, self.data, acceptance_params
        )

        fls, fl_errs = [], []
        afbs, afb_errs = [], []

        for bin_number in range(10):
            m = Minuit(df_log_likelihood, fl=0.0, afb=0.0, _bin=bin_number)
            # fixing the bin number as we don't want to optimize it
            m.fixed['_bin'] = True  
            m.limits=((0, 3.0), (-1.0, 1.0), None)
            m.migrad() #find min using gradient descent
            m.hesse() #finds estimation of errors at min point

            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])

        return fl_afb_fit(fls, fl_errs, afbs, afb_errs, [
            df_log_likelihood(fl, afb, bin) for fl, afb, bin in zip(fls,afbs,range(10))
        ])
        

    def make_and_fit_pseudoexperiments_acceptance(self,pseudodatasets):
        self.pseudoexperiments = [
            Experiment(data) for data in pseudodatasets
        ]
        for experiment in self.pseudoexperiments:
            pass

if __name__ == '__main__':
    DIR = './pseudoexperiments'
    FILE = os.path.join(DIR,'pseudoexperiments-manualselector.pkl')
    ensure_dir(DIR)

    total = load_file(RAWFILES.TOTAL_DATASET)
    s, ns = selection_all(total)

    pseudodatasets = cached_compute_pseudoexperiments(s, FILE)
    
    with open('tmp/acceptance_coeff.pkl', 'rb') as f:
        params_dict = pickle.load(f)

    exp = Experiment(s)
    exp.make_and_fit_pseudoexperiments_acceptance(pseudodatasets)