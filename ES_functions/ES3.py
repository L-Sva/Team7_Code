#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:19:22 2022

@author: ES3
"""

#%%
'''
This file includes the selection function of B vertex chi2
Test can be run if one makes sure that the pickle data file has path "data/total_dataset.pkl"
'''
#%%
import pandas as pd
import scipy as sp 
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import fsolve
#%% Selection Functions
def B0_vertex_chi2_select(dataframe, alpha):
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

# %% Test
total_dataset = pd.read_pickle('data/total_dataset.pkl')

#Chi-2 selection (reject chi probabilities less tan 20%)
select, reject = B0_vertex_chi2_select(total_dataset, 0.2)
plt.figure(1)   #Histogram of selected chi2
select['B0_ENDVERTEX_CHI2'].hist(bins=20)
plt.title('Selected')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

plt.figure(2) #histogram of rejected chi2
reject['B0_ENDVERTEX_CHI2'].hist(bins=20)
plt.title('Rejected')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

plt.figure(3) #histogram of original chi2
total_dataset['B0_ENDVERTEX_CHI2'].hist(bins=20)
plt.title('Original Dataset')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

# %% Test

#Test of the final state particle IP selection 
#Chi^2 values that give probability of particle originated form the PV less than 0.01 are retained
total_dataset = pd.read_pickle('data/total_dataset.pkl')
select, reject = final_state_particle_IP(total_dataset, 0.01)

plt.figure(1)   #Histogram of selected chi2
select['mu_plus_IPCHI2_OWNPV'].hist(bins=100)
plt.title('Selected')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

plt.figure(2) #histogram of rejected chi2
reject['mu_plus_IPCHI2_OWNPV'].hist(bins=100)
plt.title('Rejected')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

plt.figure(3) #histogram of original chi2
total_dataset['mu_plus_IPCHI2_OWNPV'].hist(bins=100)
plt.title('Original Dataset')
plt.xlabel('Chi-2')
plt.ylabel('Number of Events')

# Can see that chi2 less than ~13 are rejected 
# %%