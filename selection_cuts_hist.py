from core import load_file, RAWFILES
import pandas as pd
from ES_functions.Compiled import q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all
import numpy as np
from test_candidates_example import test_candidate_true_false_positive_negative
from histrogram_plots import generic_selector_plot_new
import matplotlib.pyplot as plt
#%%
def ml_combine_signal_bk(signal_dataset, background_dataset):
    """Combines signal and background dataset, adding category labels
    """
    # Marek
    signal_dataset = signal_dataset.copy()
    background_dataset = background_dataset.copy()
    signal_dataset.loc[:,'category'] = 1
    background_dataset.loc[:,'category'] = 0

    # combine
    dataset = pd.concat((signal_dataset, background_dataset))
    return dataset
#%%
peaking_bk=[]
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    peaking_bk.append(data)

peaking_bk_combined=pd.concat(peaking_bk)
peaking_bk.append(peaking_bk_combined)

mod_peaking_bk=[]

signal = load_file(RAWFILES.SIGNAL)
for non_signal in peaking_bk:
    test_data=ml_combine_signal_bk(signal, non_signal)
    mod_peaking_bk.append(test_data)

funclist=[q2_resonances, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID, selection_all]
funclistnames=["q2_resonances", "Kstar_inv_mass", "B0_vertex_chi2", "final_state_particle_IP", "B0_IP_chi2", "flight distance", "DIRA", "particle ID", "compiled"]
funclistnamesfull=["q\u00B2 resonances", "K*\u2070 invariant mass", "B\u2070 vertex χ\u00B2", "final state particle impact parameter", "B\u2070 impact parameter χ\u00B2", "flight distance", "DIRA angle", "particle ID", "compiled"]
peaking_bknames = ["JPSI", "JPSI_MU_K_SWAP", "JPSI_MU_PI_SWAP", "K_PI_SWAP","PHIMUMU", "PKMUMU_PI_TO_P", "PKMUMU_PI_TO_K_K_TO_P", "PSI2S","COMBINED"]
peaking_bknamesfull=["B\u2070→J/ψK*\u2070 peaking background","B\u2070→J/ψK*\u2070 \u03BC \u27F7 K peaking background","B\u2070→J/ψK*\u2070 \u03BC \u27F7 \u03C0 peaking background","B\u2070→K*\u2070μ\u207Aμ\u207B K \u27F7 \u03C0 peaking background","B\u209B\u2070→ϕμμ K \u27F7 \u03C0 peaking background","Λ_b\u2070→pKμμ p \u2192 K, K \u2192 \u03C0 peaking background","Λ_b\u2070→pKμμ p \u27F7 \u03C0 peaking background","B\u2070→J/ψ(2S)K*\u2070 peaking background","all peaking backgrounds"]
# =============================================================================
# J/ψ→μμ
# J/𝜓→𝜇𝜇
#  
# B0→J/ψK∗0
# B0→J/𝜓K∗0
#  
# B0s→ϕμμ
# Bs0→𝜙𝜇𝜇
#  
# 
# Λ0b→pKμμ
# Λb0→pK𝜇𝜇
#  
# =============================================================================

#peaking_bknamesfull=["J/\u03C8","J/\u03C8 \u03BC \u27F7 K","J/\u03C8 \u03BC \u27F7 \u03C0","K \u03BC \u03C0","\u03C6 \u03BC \u03BC","B0→J/ψK∗0"]


#%%
# =============================================================================
# #example
# column = 'B0_MM'
# 
# for selection_method in range((len(funclist)):
#     for test_data in range(len(mod_peaking_bk)):
#         #need to define column for each plot!!
#         plt.close()
#         s, ns = funclist[selection_method](mod_peaking_bk[test_data])
#         val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
#         print(val)
#         plt.figure(test_data)
#         generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = True)
#         plt.title(f'Events accepted and rejected for {peaking_bknames[test_data]} \n by {funclistnames[selection_method]} selection cut')
#         #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")
# =============================================================================
#%%

column='q2'
selection_method=0#q2 cut
for test_data in range(len(mod_peaking_bk)):
    #need to define column for each plot!!
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False, columnname='q\u00B2')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
#column='B0_FDCHI2_OWNPV'
column='B0_FD_OWNPV'
selection_method=5#flight distance
for test_data in range(len(mod_peaking_bk)):
    #need to define column for each plot!!
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='B\u2070 Flight Distance')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
# =============================================================================
# column='B0_DIRA_OWNPV'
# selection_method=6#DIRA (angle)
# for test_data in range(len(mod_peaking_bk)):
#     #need to define column for each plot!!
#     plt.close()
#     s, ns = funclist[selection_method](mod_peaking_bk[test_data])
#     val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
#     print(val)
#     plt.figure(test_data)
#     generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False)
#     plt.title(f'Events accepted and rejected by {funclistnames[selection_method]} selection cut \n for {peaking_bknamesfull[test_data]}')
#     plt.show()
#     #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")
# 
# =============================================================================
#%%
# =============================================================================
# column='q2'
# selection_method=8#combined
# for test_data in range(len(mod_peaking_bk)):
#     #need to define column for each plot!!
#     plt.close()
#     s, ns = funclist[selection_method](mod_peaking_bk[test_data])
#     val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
#     print(val)
#     plt.figure(test_data)
#     generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False)
#     plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for {peaking_bknamesfull[test_data]}')
#     plt.show()
#     #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")
# 
# =============================================================================
#%%
column='K_MC15TuneV1_ProbNNpi'#can be any of the mc15 columns - because these columns are the columns repsonsible for misidentification
selection_method=7#particle ID
for test_data in range(len(mod_peaking_bk)):
    #need to define column for each plot!!
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='Probability that K is identified as \u03C0')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
column='Kstar_MM'#can be any of the mc15 columns - because these columns are the columns repsonsible for misidentification
selection_method=1#kstar inv mass
for test_data in range(len(mod_peaking_bk)):
    #need to define column for each plot!!
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='K* Measured Mass')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")
