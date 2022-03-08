#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:56:18 2022

@author: emirsezik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.special import legendre

from modifiedselectioncuts import selection_all
from ml_selector import remove_combinatorial_background

plt.rcParams['font.size'] = 18

Path('../tmp/').mkdir(exist_ok=True)

def P_l(n, x):
    coeff = legendre(n)
    p = np.poly1d(coeff)
    return p(x)

def acceptance_function(q2, ctl, ctk, phi, coeff_q2range):
    coeff, qrange = coeff_q2range
    q2min = qrange[0]
    q2max = qrange[1]
    q2 = 2 * (np.array(dataframe["q2"]) - (q2max + q2min) / 2)/ (q2max - q2min)
    phi = phi/np.pi
    x,y,z,w = np.meshgrid(q2, ctl, ctk, phi)
    acc_func = np.zeros(x.shape)
    i_max, j_max, k_max, l_max = coeff.shape
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                for l in range(l_max):
                    acc_func += coeff[i,j,k,l] * P_l(i, x) * P_l(j, y) * P_l(k, z) * P_l(l, w)
    return acc_func

def coeff_q2range(dataframe, q2min, q2max):
    q2 = dataframe['q2']
    crit_a = (q2 >= q2min) & (q2 <= q2max)
    dataframe = dataframe[crit_a]
    N = len(dataframe)
    q2 = 2 * (np.array(dataframe["q2"]) - (q2max + q2min) / 2)/ (q2max - q2min)
    ctl = np.array(dataframe["costhetal"])
    ctk = np.array(dataframe["costhetak"])
    phi = (np.array(dataframe["phi"]))/np.pi 
    shape = (6, 5, 6, 7)
    c = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    c[i,j,k,l] = 1/(16 * N) * (2 * i + 1) * (2 * j + 1) * (2 * k + 1) * (2 * l + 1) * sum( P_l(i, q2) * P_l(j, ctl) * P_l(k, ctk) * P_l(l, phi) )
    return c, q2min, q2max

#%%
# filter acceptance_mc
dataframe = pd.read_pickle("data/acceptance_mc.pkl")
dataframe, _ = selection_all(dataframe)
dataframe, _ = remove_combinatorial_background(dataframe)
dataframe.to_pickle('../tmp/filtered_acc.pkl')

#%%
dataframe = pd.read_pickle('../tmp/filtered_acc.pkl')

#%%
c, c_q2min, c_q2max = coeff_q2range(dataframe, 0.1, 8)
np.save('../tmp/coeff.npy', c)

#%%
c_q2min, c_q2max = 0.1, 8
c = np.load('../tmp/coeff.npy')

#All the following are testing
#%%
q2 = dataframe['q2']
crit_a = (q2 >= c_q2min) & (q2 <= c_q2max)
dataframe = dataframe[crit_a]

#%% check q2
bins = plt.hist(dataframe["q2"], bins = 100)[0]

q2_range = np.linspace(0.1, 8, 1000)
def func(q2, q2min, q2max):
    q2 = 2 * (q2 - (q2max + q2min) / 2)/ (q2max - q2min)
    num = 0
    for i in range(len(c)):
        num += P_l(i, q2) * c[i, 0, 0, 0]
    return num
acc = [func(i, 0.1, 8) for i in q2_range]
plt.plot(q2_range, np.array(acc)* bins[0] / func(0.1, 0.1, 8)*0.95 )
params={
   'axes.labelsize': 30,
   'font.size': 30,
   'legend.fontsize': 30,
   'xtick.labelsize': 30,
   'ytick.labelsize': 30,
   'figure.figsize': [16, 9]
   } 
plt.rcParams.update(params)
plt.xlabel('q2')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()

#%% check ctl

ctl_bins = plt.hist(dataframe["costhetal"], bins = 100)[0]

ctl_range = np.linspace(-1, 1, 1000)
def func2(ctl):
    num = 0
    for i in range(len(c[0])):
        num += P_l(i, ctl) * c[0, i, 0, 0]
    return num
acc = np.array([func2(i) for i in ctl_range])
plt.plot(ctl_range, acc * ctl_bins[0] / func2(-1)*0.9)
plt.xlabel('ctl')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()
plt.show()

#%% check ctk
ctk_bins = plt.hist((dataframe)["costhetak"], bins = 100)[0]

ctk_range = np.linspace(-1, 1, 1000)
def func3(ctk):
    num = 0
    for i in range(len(c[0, 0])):
        num += P_l(i, ctk) * c[0, 0, i, 0]
    return num
acc = np.array([func3(i) for i in ctk_range])
plt.plot(ctk_range, acc * ctk_bins[0] / func3(-1))
plt.xlabel('ctk')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()
plt.show()

#%% check phi
phi_bins = plt.hist((dataframe)["phi"], bins = 100)[0]

phi_range = np.linspace(-np.pi, np.pi, 1000)
def func1(phi):
    phi = phi/np.pi
    num = 0
    for i in range(len(c[0, 0, 0])):
        num += P_l(i, phi) * c[0, 0, 0, i]
    return num
acc = np.array([func1(i) for i in phi_range])
plt.plot(phi_range, acc * phi_bins[0] / func1(-np.pi))
plt.xlabel('phi')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()
plt.show()
