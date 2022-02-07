#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:24:13 2022

@author: hukaiyu
"""

# Checking the example selector in core works

import pandas as pd
import numpy as np
from core import load_file, B0_MM_selector, save_file
import matplotlib.pyplot as plt

# Allows file to be loaded as a module to allow reuse of functions defined above this line
# Do not define gloabal vars above this line
if __name__ == '__main__':

    # Load the datasets
    total_dataset = load_file('/Users/antheaml/Desktop/University/Year 3 Courses/Term 2/TBPS/Team7_Code-main/data/total_dataset.pkl')
#%%
    mu_plus_threshold = 0.9


    #first selection for mu_plus   
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['mu_plus_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['mu_plus_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['mu_plus_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['mu_plus_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['mu_plus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n3 and n3 > mu_plus_threshold): # make sure prob. that a particle identified as muon is a real moun is the maximum
            prob_real.append(n3)
    
    surv_num = len(prob_real)
    surv_frac = (len(prob_real)/len(total_dataset))
    fig = plt.figure(figsize = (11.0, 8.0))
    plt.hist(prob_real,bins=50)
    plt.xlabel(r'Prob. a particle identified as $\mu\plus$ is correctly identified as a $\mu$', fontsize = 15) 
    plt.ylabel('Frequency', fontsize = 15)
    plt.title(r'$\mu\plus$, 50 bins, threshold = {mu_plus_threshold}, surviving number = {surv_num} , surviving fraction: = {surv_frac} '.format(mu_plus_threshold = mu_plus_threshold, surv_num = surv_num, surv_frac = round(surv_frac, 4)), fontsize = 15)
    print('Surviving events:', len(prob_real))

#%%
    mu_minus_threshold = 0.99
    #first selection for mu_minus   
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['mu_minus_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['mu_minus_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['mu_minus_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['mu_minus_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['mu_minus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n3 and n3 > mu_minus_threshold): # make sure prob. that a particle identified as muon is a real moun is the maximum
            prob_real.append(n3)
            
    surv_num = len(prob_real)
    surv_frac = (len(prob_real)/len(total_dataset))
    fig = plt.figure(figsize = (18.0, 10.0))
    plt.hist(prob_real,bins=50)
    plt.xlabel(r'Prob. a particle identified as $\mu\minus$ is correctly identified as a $\mu$', fontsize = 15) 
    plt.ylabel('Frequency', fontsize = 15)
    plt.title(r'$\mu\minus$, 50 bins, threshold = {mu_minus_threshold}, surviving number = {surv_num} , surviving fraction: = {surv_frac} '.format(mu_minus_threshold = mu_minus_threshold, surv_num = surv_num, surv_frac = round(surv_frac, 4)), fontsize = 15)
    fig.set_size_inches(10, 8) 
    print(surv_num, surv_frac)                                         
    
#%%
    k_threshold = 0.99
    #first selection for Kaon  
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['K_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['K_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['K_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['K_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['K_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n1 and n1 > k_threshold): # make sure prob. that a particle identified as kaon is a real kaon is the maximum
            prob_real.append(n1)
            
    
    surv_num = len(prob_real)
    surv_frac = (len(prob_real)/len(total_dataset))
    fig = plt.figure(figsize = (18.0, 10.0))
    plt.hist(prob_real,bins=50)
    plt.xlabel(r'Prob. a particle identified as $K$ is correctly identified as a $K$', fontsize = 15) 
    plt.ylabel('Frequency', fontsize = 15)
    plt.title(r'$K$, 50 bins, threshold = {k_threshold}, surviving number = {surv_num} , surviving fraction: = {surv_frac} '.format(k_threshold = k_threshold, surv_num = surv_num, surv_frac = round(surv_frac, 4)), fontsize = 15)
    fig.set_size_inches(10, 8) 
    print(surv_num, surv_frac)
    
#%%
    
    pi_threshold = 0.99
    #first selection for Pion 
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['Pi_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['Pi_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['Pi_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['Pi_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['Pi_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n2 and n2 > pi_threshold): # make sure prob. that a particle identified as pion is a real pion is the maximum
            prob_real.append(n2)
    
    surv_num = len(prob_real)
    surv_frac = (len(prob_real)/len(total_dataset))
    fig = plt.figure(figsize = (18.0, 10.0))
    plt.hist(prob_real,bins=50)
    plt.xlabel(r'Prob. a particle identified as $\pi$ is correctly identified as a $\pi$', fontsize = 15) 
    plt.ylabel('Frequency', fontsize = 15)
    plt.title(r'$\pi$, 50 bins, threshold = {pi_threshold}, surviving number = {surv_num} , surviving fraction: = {surv_frac} '.format(pi_threshold = pi_threshold, surv_num = surv_num, surv_frac = round(surv_frac, 4)), fontsize = 15)
    fig.set_size_inches(10, 8) 
    print('Surviving events:', len(prob_real))





    