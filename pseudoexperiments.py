import os
import pickle
from ES_functions.Compiled import selection_all
from core import RAWFILES, ensure_dir, load_file
from ml_selector import remove_all_bk

def compute_pseudoexperiments(data):
    pseudoexperiments = [s.sample(frac=1, replace=True, axis=0) for i in range(2000)]
    return pseudoexperiments

if __name__ == '__main__':
    DIR = './pseudoexperiments'
    FILE = os.path.join(DIR,'pseudoexperiments-manualselector.pkl')
    ensure_dir(DIR)

    if not os.path.exists(FILE):
        total = load_file(RAWFILES.TOTAL_DATASET)

        s, ns = selection_all(total)

        pseudoexperiments = compute_pseudoexperiments(s)

        with open(FILE, 'wb') as file:
            pickle.dump(pseudoexperiments, file)
    else:
        with open(FILE, 'wb') as file:
            pseudoexperiments = pickle.load(file)

    with open('tmp/acceptance_coeff.pkl', 'rb') as f:
        params_dict = pickle.load(f)

    