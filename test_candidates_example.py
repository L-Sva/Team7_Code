from core import load_file, RAWFILES
from ml_tools import test_false_true_negative_positive, test_sb
import ml_tools
import pandas as pd
from ES_functions.Compiled import q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all
import ml_load
import numpy as np

#%%
def test_candidate_true_false_positive_negative(test_data, selection_method=q2_resonances):
    
    s, ns = selection_method(test_data)
    s = s.copy()
    ns = ns.copy()
    s['pred'] = 1
    ns['pred'] = 0
    test_res_data = pd.concat((s,ns))
    sig_prob = test_res_data['pred'].to_numpy()
    
    dict1=test_false_true_negative_positive(test_res_data, sig_prob ,0.5)
    dict2={'SB Metric':test_sb(test_res_data, sig_prob ,0.5)}#find SB metric and make it a dictionary
    both_dicts = {**dict1, **dict2}#merge dictionaries
    return both_dicts
    
#%%
#Load signal and non-signal, then loop through selection cuts including "selection_all" which is all cuts
signal = load_file(RAWFILES.SIGNAL)
#signal=pd.read_pickle('data/signal.pkl')

non_signal = []
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    #data = pd.read_pickle(f'data/{file}')
    non_signal.append(data)
non_signal = pd.concat(non_signal)
test_data = ml_tools.ml_combine_signal_bk(signal, non_signal)

train, test = ml_load.get_train_test_for_all_peaking_bks()

funclist=[q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA,] #Particle_ID, selection_all]
output={}
for func in funclist:
    print('Testing selector', func.__name__)
    output[func.__name__] = (test_candidate_true_false_positive_negative(test,func))
for key in output:
    print(f'{key} | tpr: {output[key]["true-positive"]}, fpr: {output[key]["false-positive"]}, sb: {output[key]["SB Metric"]}')
#30/01/2022: Error with Particle_ID func (and therefore selection_all func) - possibly due to ml_combine_signal_bk func used in line 9?