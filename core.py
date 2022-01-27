from dataclasses import dataclass
import pandas as pd
import os
import os.path as path
import pickle

# Defining file name references to core data files (ide can autofill these)
@dataclass
class RAWFILES:
    TOTAL_DATASET = 'total_dataset.pkl'
    JPSI = 'jpsi.pkl'
    SIGNAL = 'signal.pkl'
    JPSI_MU_K_SWAP = 'jpsi_mu_k_swap.pkl'
    JPSI_MU_PI_SWAP = 'jpsi_mu_pi_swap.pkl'
    K_PI_SWAP = 'k_pi_swap.pkl'
    PHIMUMU = 'phimumu.pkl'
    PKMUMU_PI_TO_P = 'pKmumu_piTop.pkl'
    PKMUMU_PI_TO_K_K_TO_P = 'pKmumu_piTok_kTop.pkl'
    PSI2S = 'psi2S.pkl'
    ACCEPTANCE = 'acceptance_mc.pkl'
    peaking_bks = [JPSI, JPSI_MU_K_SWAP, JPSI_MU_PI_SWAP, K_PI_SWAP,
        PHIMUMU, PKMUMU_PI_TO_P, PKMUMU_PI_TO_K_K_TO_P, PSI2S]
    
def load_file(filename='total_dataset.pkl', folder='data') -> pd.DataFrame:
    res = None
    with open(path.join(folder,filename),'rb') as file:
        res = pickle.load(file)
    return res

def save_file(dataframe ,filename, foldername):
    if not path.exists(foldername):
        os.mkdir(foldername)
    with open(path.join(foldername, filename),'wb') as file:
        pickle.dump(dataframe,file)
    
def B0_MM_selector(dataframe):
    subset = dataframe[dataframe['B0_MM'] < 5350]
    not_subset = dataframe[dataframe['B0_MM'] > 5350]
    return subset, not_subset

def combine_n_selectors(*selectors):
    def combined_selectors(data_set, **kwargs):
        expected = []
        for selector in selectors:
            expected.append([kwargs[key] for key in (selector.__name__,) if key in kwargs])
        no = []
        yes = data_set
        for i, selector in enumerate(selectors):
            yes, no_sel = selector(yes, *expected[i])
            no.append(no_sel)
            if len(yes) < 100:
                raise ValueError('len')
        return yes, pd.concat(no)
    return combined_selectors

# selector - output as additional column instead of pair of subsets