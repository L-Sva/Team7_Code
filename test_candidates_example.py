from core import load_file, RAWFILES
import pandas as pd
from ES_functions.Compiled import q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all
# import ml_load
import numpy as np

#%%
def test_false_true_negative_positive(test_dataset, sig_prob, threshold) -> dict:
    # Jiayang

    x = test_dataset['category'].to_numpy()

    x_mask_0 = x == 0
    x_mask_1 = x == 1
    prb_mask_pos = sig_prob >= threshold
    prb_mask_neg = sig_prob < threshold

    signal = np.count_nonzero(x_mask_1)
    background = np.count_nonzero(x_mask_0)
    true_positive =  np.count_nonzero(np.logical_and(x_mask_1, prb_mask_pos))
    false_negative = np.count_nonzero(np.logical_and(x_mask_1, prb_mask_neg))
    false_positive = np.count_nonzero(np.logical_and(x_mask_0, prb_mask_pos))
    true_negative =  np.count_nonzero(np.logical_and(x_mask_0, prb_mask_neg))
    
    # rates
    tpr = true_positive / signal
    fpr = false_positive / background

    fnr = false_negative / signal
    tnr = true_negative / background

    return {
        'true-positive': tpr,
        'false-positive': fpr,
        'true-negative': tnr,
        'false-negative': fnr,
        'signal': signal,
        'background': background,
        'n-signal-accept': signal * tpr,
        'n-background-accept': background * fpr,
    }

def test_sb(test_dataset, sig_prob, threshold):
    # Jiayang

    output = test_false_true_negative_positive(test_dataset, sig_prob, threshold)

    S = output['signal'] * output['true-positive']
    B = output['background'] * output['false-positive']
    if S+B == 0:
        return 0
    metric = S/np.sqrt(S+B)
    return metric

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
if __name__ == '__main__':
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
        print(f'{key} | tpr: {output[key]["true-positive"]:.4g}, fpr: {output[key]["false-positive"]:.4g}, sb: {output[key]["SB Metric"]:4g}')
    #30/01/2022: Error with Particle_ID func (and therefore selection_all func) - possibly due to ml_combine_signal_bk func used in line 9?