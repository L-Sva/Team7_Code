from core import load_file, RAWFILES
from ml_tools import test_false_true_negative_positive, test_sb
import ml_tools
import pandas as pd
from ES_functions.Compiled import q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all

#%%
def test_candidate_true_false_positive_negative(signal, non_signal, selection_method=q2_resonances):
    dataframe = ml_tools.ml_combine_signal_bk(signal, non_signal)
    s, ns = selection_method(dataframe)
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

funclist=[q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all]
dictlist=[]
for func in funclist:
    dictlist.append(test_candidate_true_false_positive_negative(signal, non_signal,func))
print(dictlist)
#30/01/2022: Error with Particle_ID func (and therefore selection_all func) - possibly due to ml_combine_signal_bk func used in line 9?