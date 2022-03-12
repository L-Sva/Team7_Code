from core import load_file, RAWFILES
import pandas as pd
from ES_functions.modifiedselectioncuts import q2_resonances,q2_range,K0_vertex_chi2, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD,KSTAR_FD, DIRA, Particle_ID,B0_mass, selection_all,peaking_back_1,peaking_back_2,peaking_back_3,peaking_back_4,peaking_back_5,peaking_back_6,peaking_back_7,peaking_back_8
import numpy as np
from test_candidates_example import test_candidate_true_false_positive_negative
from histrogram_plots_1 import generic_selector_plot_new
import matplotlib.pyplot as plt
#%%
#Function combines signal and peaking background events
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
#Load peaking backgrounds
peaking_bk=[]
for file in RAWFILES.peaking_bks:
    data = load_file(file)
    peaking_bk.append(data)
#create a "combined peaking background" background by concatenating invidual ones
peaking_bk_combined=pd.concat(peaking_bk)
peaking_bk.append(peaking_bk_combined)

mod_peaking_bk=[]
#combine the backgrounds with the signal
signal = load_file(RAWFILES.SIGNAL)
for non_signal in peaking_bk:
    test_data=ml_combine_signal_bk(signal, non_signal)
    mod_peaking_bk.append(test_data)
#list of selection cuts, their names, peaking backgrounds, their names
funclist=[q2_resonances,q2_range,K0_vertex_chi2, Kstar_inv_mass, B0_vertex_chi2, final_state_particle_IP, B0_IP_chi2, FD,KSTAR_FD, DIRA, Particle_ID,B0_mass, selection_all,peaking_back_1,peaking_back_2,peaking_back_3,peaking_back_4,peaking_back_5,peaking_back_6,peaking_back_7,peaking_back_8]
#funclistnames=["q2_resonances", "Kstar_inv_mass", "B0_vertex_chi2", "final_state_particle_IP", "B0_IP_chi2", "flight distance", "DIRA", "particle ID", "compiled"]
funclistnamesfull=["q\u00B2 resonances", "q\u00B2 range", "K*\u2070 vertex χ\u00B2","K*\u2070 invariant mass", "B\u2070 vertex χ\u00B2", "final state particle impact parameter", "B\u2070 impact parameter χ\u00B2", "B\u2070 flight distance","K* flight distance", "DIRA angle", "particle ID","B0 mass", "compiled", "peaking background 1", "peaking background 2", "peaking background 3", "peaking background 4", "peaking background 5", "peaking background 6", "peaking background 7", "peaking background 8"]
#peaking_bknames = ["JPSI", "JPSI_MU_K_SWAP", "JPSI_MU_PI_SWAP", "K_PI_SWAP","PHIMUMU", "PKMUMU_PI_TO_P", "PKMUMU_PI_TO_K_K_TO_P", "PSI2S","COMBINED"]
peaking_bknamesfull=["B\u2070→J/ψK*\u2070 peaking background","B\u2070→J/ψK*\u2070 \u03BC \u27F7 K peaking background","B\u2070→J/ψK*\u2070 \u03BC \u27F7 \u03C0 peaking background","B\u2070→K*\u2070μ\u207Aμ\u207B K \u27F7 \u03C0 peaking background","B\u209B\u2070→ϕμμ K \u27F7 \u03C0 peaking background","Λ_b\u2070→pKμμ p \u2192 K, K \u2192 \u03C0 peaking background","Λ_b\u2070→pKμμ p \u27F7 \u03C0 peaking background","B\u2070→J/ψ(2S)K*\u2070 peaking background","all peaking backgrounds"]

#%%
#q2 resonances cut on q2 column
column='q2'
selection_method=0#q2 resonances cut
for test_data in range(len(mod_peaking_bk)):
    #need to define column for each plot!!
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data,dpi=200)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False, columnname='q\u00B2')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
#B0 flight distance cut on flight distance column
column='B0_FDCHI2_OWNPV'
#column='B0_FD_OWNPV'
selection_method=7#B0 flight distance
for test_data in range(len(mod_peaking_bk)):
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data,dpi=200)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='B\u2070 Flight Distance χ\u00B2 w.r.t Primary Vertex')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
#DIRA cut produces poor graphs because the angle that we cut at is so small
# =============================================================================
# column='B0_DIRA_OWNPV'
# selection_method=9#DIRA (angle)
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
##combined selection cut
# =============================================================================
# column='q2'
# selection_method=11#combined
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
#Particle ID selection cut performed on "probability that a Kaon is identified as a pion" column.
column='K_MC15TuneV1_ProbNNpi'#can be another mc15 column - because these columns are the columns repsonsible for misidentification
selection_method=10#particle ID
for test_data in range(len(mod_peaking_bk)):
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data,dpi=200)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='Probability that K is identified as \u03C0')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")

#%%
column='Kstar_MM'#can be any of the mc15 columns - because these columns are the columns repsonsible for misidentification
selection_method=3#kstar inv mass
for test_data in range(len(mod_peaking_bk)):
    plt.close()
    s, ns = funclist[selection_method](mod_peaking_bk[test_data])
    val = test_candidate_true_false_positive_negative(mod_peaking_bk[test_data], selection_method = funclist[selection_method])
    print(val)
    plt.figure(test_data,dpi=200)
    generic_selector_plot_new(orginal = mod_peaking_bk[test_data],subset = s, not_subset = ns, column = column, bins = 100, show = False,columnname='K* Measured Mass')
    plt.title(f'Events accepted and rejected by {funclistnamesfull[selection_method]} selection cut \n for signal + {peaking_bknamesfull[test_data]}')
    plt.show()
    #plt.savefig(f"Selection_cuts_histograms_q2/{column}_{funclistnames[selection_method]}_{peaking_bknames[test_data]}.png")
