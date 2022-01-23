#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:24:13 2022

@author: hukaiyu
"""

# Checking the example selector in core works

import pandas as pd
import numpy as np
from core import load_file, example_selector, save_file
import matplotlib.pyplot as plt

# Allows file to be loaded as a module to allow reuse of functions defined above this line
# Do not define gloabal vars above this line
if __name__ == '__main__':

    # Load the datasets
    total_dataset = load_file('/Users/hukaiyu/Desktop/Y3/TBPS/total_dataset.pkl')

#%%
    #first selection for mu_plus   
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['mu_plus_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['mu_plus_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['mu_plus_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['mu_plus_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['mu_plus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n3): # make sure prob. that a particle identified as muon is a real moun is the maximum
            prob_real.append(n3)
    
    plt.hist(prob_real,bins=50)
    plt.xlabel('probability that a particle identified as muon_plus is a real moun after first selection') 
    plt.ylabel('number of occurrences')
    plt.title('mu_plus')   

#%%
    #first selection for mu_minus   
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['mu_minus_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['mu_minus_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['mu_minus_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['mu_minus_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['mu_minus_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n3): # make sure prob. that a particle identified as muon is a real moun is the maximum
            prob_real.append(n3)
    
    plt.hist(prob_real,bins=50)
    plt.xlabel('probability that a particle identified as muon_minus is a real moun after first selection') 
    plt.ylabel('number of occurrences')
    plt.title('mu_minus')                                             
    
#%%
    #first selection for Kaon  
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['K_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['K_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['K_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['K_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['K_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n1): # make sure prob. that a particle identified as kaon is a real kaon is the maximum
            prob_real.append(n1)
    
    plt.hist(prob_real,bins=50)
    plt.xlabel('probability that a particle identified as Kaon is a real Kaon after first selection') 
    plt.ylabel('number of occurrences')
    plt.title('Kaon') 
    
#%%
    #first selection for Pion 
    prob_real = []
    
    for i in range (len(total_dataset['mu_plus_MC15TuneV1_ProbNNk'])):
        
        n1 = total_dataset['Pi_MC15TuneV1_ProbNNk'][i]
        n2 = total_dataset['Pi_MC15TuneV1_ProbNNpi'][i]
        n3 = total_dataset['Pi_MC15TuneV1_ProbNNmu'][i]
        n4 = total_dataset['Pi_MC15TuneV1_ProbNNe'][i]
        n5 = total_dataset['Pi_MC15TuneV1_ProbNNp'][i]
                                                         
        maximum = max(n1,n2,n3,n4,n5)

        if (maximum == n2): # make sure prob. that a particle identified as pion is a real pion is the maximum
            prob_real.append(n2)
    
    plt.hist(prob_real,bins=50)
    plt.xlabel('probability that a particle identified as Pion is a real Pion after first selection') 
    plt.ylabel('number of occurrences')
    plt.title('Pion')     