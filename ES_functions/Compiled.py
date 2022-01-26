#%%
import numpy as np 
import scipy as sp 
from scipy.optimize import fsolve
import scipy.stats as ss
import pandas as pd

# %% List of all selection functions
def q2_resonances(data):
    q2 = data['q2']
    crit_a = (q2 > 8) & (q2 < 11) #criteria A
    crit_b = (q2 > 12.5) & (q2 < 15) #criteria B
    subset = data[~crit_a & ~crit_b] #not crit_a and not crit_b
    not_subset = data[crit_a | crit_b] # crit_a or crit_b
    return subset, not_subset

def Kstar_inv_mass(dataframe):
    subset = []
    not_subset = []
    Kstar_MM = np.array(dataframe["Kstar_MM"])
    for i in range(len(Kstar_MM)):
        if (795.9 < Kstar_MM[i]) and (Kstar_MM[i] < 995.9): #in MeV
            subset.append(i)
            continue
        else:
            not_subset.append(i)

    subset = dataframe.iloc[subset]

    not_subset = dataframe.iloc[not_subset]

    return subset, not_subset
def B0_vertex_chi2(dataframe, alpha):
    '''
    Input: dataframe - a dataframe need to be processed
            alpha - (in range 0 to 1) threshold of the propability of obtaining 
                    such a chi2; events with a large chi2 such that the probability
                    (of obtaining a chi2 as large or even larger) lower than alpha will be rejected.

    Output: subset - selected candidates
            not_subset - rejected candidates
    '''
    def func(x):
        return ss.chi2.sf(x,5) - alpha    # scipy.chi2.sf() gives the survival function of Chi2 distribution, 5 degrees of freedom for vertex
    threshold = float(fsolve(func, 5.))   #These 3 lines solves for the threshold chi2 value given the threshold alpha (probability)

    subset = dataframe[dataframe['B0_ENDVERTEX_CHI2'] <= threshold]
    not_subset = dataframe[dataframe['B0_ENDVERTEX_CHI2'] > threshold] 
    return subset, not_subset

def final_state_particle_IP(dataframe, alpha):
    '''
    Input: dataframe - dataframe need to be processed
    alpha - (0 to 1) probability threshold: chi2 values that give probability
            lower than this threshold are selected (high chi2, low probability of 
            being a particle form the PV).
    
    Output: subset - selected candidates
            not_subset - rejected candidates 
    '''
    def func(x):
        return ss.chi2.sf(x,4) - alpha  #scipy.stats.chi2.sf givs the survival function of chi2 distribution
        # 4 degrees of freedom for IP (4 pieces of info in track + 3 position of PV - 3 position of closest approach)
    
    threshold = float(fsolve(func, 4.))
    subset = dataframe[(dataframe['mu_plus_IPCHI2_OWNPV'] > threshold) & \
            (dataframe['mu_minus_IPCHI2_OWNPV'] > threshold) & \
            (dataframe['K_IPCHI2_OWNPV'] > threshold) & \
            (dataframe['Pi_IPCHI2_OWNPV'] > threshold)]
    mu_plus = dataframe[dataframe['mu_plus_IPCHI2_OWNPV'] <= threshold]
    mu_minus = mu_plus[mu_plus['mu_plus_IPCHI2_OWNPV'] <= threshold]
    K = mu_minus[mu_minus['K_IPCHI2_OWNPV'] <= threshold]
    not_subset = K[K['Pi_IPCHI2_OWNPV'] <= threshold]
    return subset, not_subset

def B0_IP_chi2(dataframe,threshold):
    accept = dataframe[dataframe['B0_IPCHI2_OWNPV'] < threshold]
    reject = dataframe[dataframe['B0_IPCHI2_OWNPV'] >= threshold]
    return accept, reject

def FD(dataframe, threshold=10.):
    '''
    Input: dataframe - the dataframe need to be cleaned
            threshold - minimum flight distance accepted (default = 10.) in units of mm
    Output: subset and not_subset
    '''
    subset = dataframe[dataframe['B0_FD_OWNPV'] > threshold]
    not_subset = dataframe[dataframe['B0_FD_OWNPV'] <= threshold]
    return subset, not_subset

def DIRA(dataframe, threshold):
    subset = dataframe[dataframe['B0_DIRA_OWNPV']>threshold]
    not_subset = dataframe[dataframe['B0_DIRA_OWNPV']<=threshold]
    return subset, not_subset

def Particle_ID(dataframe_1):
    dataframe = dataframe_1.copy()
    dataframe_2 = dataframe_1.copy()
    not_candidate = set()
    for i in range (len(dataframe['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = dataframe['mu_plus_MC15TuneV1_ProbNNk'][i]
        n2 = dataframe['mu_plus_MC15TuneV1_ProbNNpi'][i]
        n3 = dataframe['mu_plus_MC15TuneV1_ProbNNmu'][i]
        n4 = dataframe['mu_plus_MC15TuneV1_ProbNNe'][i]
        n5 = dataframe['mu_plus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum != n3): # make sure prob. that a particle identified as muon is a real moun is the maximum
            not_candidate.add(i)
            
    for i in range (len(dataframe['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = dataframe['mu_minus_MC15TuneV1_ProbNNk'][i]
        n2 = dataframe['mu_minus_MC15TuneV1_ProbNNpi'][i]
        n3 = dataframe['mu_minus_MC15TuneV1_ProbNNmu'][i]
        n4 = dataframe['mu_minus_MC15TuneV1_ProbNNe'][i]
        n5 = dataframe['mu_minus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum != n3): # make sure prob. that a particle identified as muon is a real moun is the maximum
            not_candidate.add(i)
    
    for i in range (len(dataframe['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = dataframe['K_MC15TuneV1_ProbNNk'][i]
        n2 = dataframe['K_MC15TuneV1_ProbNNpi'][i]
        n3 = dataframe['K_MC15TuneV1_ProbNNmu'][i]
        n4 = dataframe['K_MC15TuneV1_ProbNNe'][i]
        n5 = dataframe['K_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum != n1): # make sure prob. that a particle identified as kaon is a real kaon is the maximum
            not_candidate.add(i)
    
    for i in range (len(dataframe['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = dataframe['Pi_MC15TuneV1_ProbNNk'][i]
        n2 = dataframe['Pi_MC15TuneV1_ProbNNpi'][i]
        n3 = dataframe['Pi_MC15TuneV1_ProbNNmu'][i]
        n4 = dataframe['Pi_MC15TuneV1_ProbNNe'][i]
        n5 = dataframe['Pi_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum != n2): # make sure prob. that a particle identified as pion is a real pion is the maximum
            not_candidate.add(i)
    
    not_candidate = sorted(list(not_candidate))
    subset = dataframe.drop(not_candidate)

    not_subset = dataframe_2.drop(subset.index)
    return subset, not_subset
#%% Selection Criteria ALL
def selection_all(dataframe, B0_vertex_prob_threshold=0.2, \
    final_particle_prob_threshold=0.02, B0_IP_chi2_threshold=9., \
        B0_FD_threshold=10., DIRA_threshold=0.9999999):
    yes_PID, no_PID = Particle_ID(dataframe)
    
    yes_q2, no_q2 = q2_resonances(yes_PID)
    
    yes_Kstar_mass, no_Kstar_mass = Kstar_inv_mass(yes_q2)
    
    yes_B0_vertex, no_B0_vertex = B0_vertex_chi2(yes_Kstar_mass,B0_vertex_prob_threshold)
    
    yes_B0_IP, no_B0_IP = B0_IP_chi2(yes_B0_vertex, B0_IP_chi2_threshold)
    
    yes_fs_IP, no_fs_IP = final_state_particle_IP(yes_B0_IP, final_particle_prob_threshold)
    
    yes_FD, no_FD = FD(yes_fs_IP, B0_FD_threshold)
    
    yes_DIRA, no_DIRA = DIRA(yes_FD, DIRA_threshold)

   
    
    no = [no_q2, no_Kstar_mass, no_B0_vertex, no_B0_IP, no_fs_IP, no_FD, no_DIRA, no_PID]
    not_subset = pd.concat(no)
    subset = yes_DIRA

    return subset, not_subset

# %% Test
import pandas as pd
total_dataset = pd.read_pickle('data/total_dataset.pkl')
selected, not_selected = selection_all(total_dataset)

# Only 216 events selected (possibly tweek thresholds)
# %%
