from core import RAWFILES
import pandas as pd
from ES_functions.Compiled import (
    q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, 
    B0_IP_chi2, FD, DIRA, Particle_ID, selection_all
)
from ml_tools import test_false_true_negative_positive, test_sb
from ml_main import load_train_validate_test, combine_signal_background, concat_datasets

def test_candidate_true_false_positive_negative(test_data, selection_method=q2_resonances):
    
    s, ns = selection_method(test_data)
    s = s.copy()
    ns = ns.copy()
    s['pred'] = 1
    ns['pred'] = 0
    test_res_data = pd.concat((s,ns))
    sig_prob = test_res_data['pred'].to_numpy()
    
    dict1=test_false_true_negative_positive(test_res_data, sig_prob ,0.5)
    #find SB metric and make it a dictionary
    dict2={'SB Metric':test_sb(test_res_data, sig_prob ,0.5)}
    #merge dictionaries
    both_dicts = {**dict1, **dict2}
    return both_dicts
    

#%%
if __name__ == '__main__':
    # Load signal and non-signal, then loop through selection cuts including 
    # "selection_all" which is all cuts
    signal = load_train_validate_test(RAWFILES.SIGNAL, validate_size=0)
    background = concat_datasets([
        load_train_validate_test(file, validate_size=0) for file in RAWFILES.peaking_bks])
    train, test = combine_signal_background(signal, background)

    funclist=[q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, 
        B0_IP_chi2, FD, DIRA, Particle_ID, selection_all]
    output={}
    for func in funclist:
        print('Testing selector', func.__name__)
        output[func.__name__] = (test_candidate_true_false_positive_negative(test,func))
    for key in output:
        print(
            (
                f'{key} | tpr: {output[key]["true-positive"]:.4g}, '
                f'fpr: {output[key]["false-positive"]:.4g}, sb: {output[key]["SB Metric"]:4g}'
            )
        )
