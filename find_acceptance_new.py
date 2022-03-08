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
from scipy.integrate import quad

from modifiedselectioncuts import selection_all, selection_all_withoutres, q2_resonances
from ml_selector import remove_combinatorial_background

plt.rcParams['font.size'] = 18

Path('../tmp/').mkdir(exist_ok=True)

def P_l(n, x):
    coeff = legendre(n)
    p = np.poly1d(coeff)
    return p(x)

def acceptance_function(q2, ctl, ctk, phi, coeff):
    q2 = (q2-9.55)/9.45
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

def coeff(dataframe):
    q2 = dataframe['q2']
    N = len(dataframe)
    q2 = (q2-9.55)/9.45
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
    return c

#%%
# filter acceptance_mc
dataframe = pd.read_pickle("data/acceptance_mc.pkl")
dataframe, _ = selection_all_withoutres(dataframe)
dataframe_with_res, _ = remove_combinatorial_background(dataframe)
dataframe_with_res.to_pickle('../tmp/filtered_acc_with_res.pkl')
#dataframe_without_res, _ = q2_resonances(dataframe_with_res)
#dataframe_without_res.to_pickle('../tmp/filtered_acc_without_res.pkl')

#%%
dataframe_with_res = pd.read_pickle('../tmp/filtered_acc_with_res.pkl')
#dataframe_without_res = pd.read_pickle('../tmp/filtered_acc_without_res.pkl')

#%%
c = coeff(dataframe_with_res)
np.save('../tmp/coeff.npy', c)

#%%
c = np.load('../tmp/coeff.npy')

#All the following are testing

# =============================================================================
# #%% check q2
# bins = plt.hist(dataframe_without_res["q2"], bins = 1000)[0]
# 
# q2_range = np.linspace(0.1, 19, 1000)
# def func(q2):
#     q2 = (q2-9.55)/9.45
#     num = 0
#     for i in range(len(c)):
#         num += P_l(i, q2) * c[i, 0, 0, 0]
#     return num
# acc = [func(i) for i in q2_range]
# plt.plot(q2_range, np.array(acc) * bins[0] / func(0.1))
# params={
#    'axes.labelsize': 30,
#    'font.size': 30,
#    'legend.fontsize': 30,
#    'xtick.labelsize': 30,
#    'ytick.labelsize': 30,
#    'figure.figsize': [16, 9]
#    } 
# plt.rcParams.update(params)
# plt.xlabel('q2')
# plt.ylabel('Counts')
# plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
# plt.show()
# 
# #%% check ctl
# ctl_range = np.linspace(-1, 1, 100)
# def func2(ctl):
#     num = 0
#     for i in range(len(c[0])):
#         
#         c_P_l= 0
#         def inte(q2):
#             q2 = (q2-9.55)/9.45
#             inte = 0
#             for j in range(len(c)):
#                 inte += P_l(j, q2) * c[j, i, 0, 0]
#             return inte
#         c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
#         
#         num += P_l(i, ctl) * c_P_l
#     return num
# acc = np.array([func2(i) for i in ctl_range])
# 
# ctl_bins = plt.hist(dataframe_without_res["costhetal"], bins = 100)[0]
# plt.plot(ctl_range, acc * ctl_bins[0] / func2(-1)*0.95)
# plt.xlabel('ctl')
# plt.ylabel('Counts')
# plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
# plt.show()
# plt.show()
# 
# #%% check ctk
# ctk_range = np.linspace(-1, 1, 100)
# def func3(ctk):
#     num = 0
#     for i in range(len(c[0, 0])):
#         
#         c_P_l= 0
#         def inte(q2):
#             q2 = (q2-9.55)/9.45
#             inte = 0
#             for j in range(len(c)):
#                 inte += P_l(j, q2) * c[j, 0, i, 0]
#             return inte
#         c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
#         
#         num += P_l(i, ctk) * c_P_l
#     return num
# acc = np.array([func3(i) for i in ctk_range])
# 
# ctk_bins = plt.hist((dataframe_without_res)["costhetak"], bins = 100)[0]
# plt.plot(ctk_range, acc * ctk_bins[0] / func3(-1))
# plt.xlabel('ctk')
# plt.ylabel('Counts')
# plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
# plt.show()
# plt.show()
# 
# #%% check phi
# phi_range = np.linspace(-np.pi, np.pi, 100)
# def func4(phi):
#     num = 0
#     phi = phi/np.pi
#     for i in range(len(c[0, 0, 0])):
#         
#         c_P_l= 0
#         def inte(q2):
#             q2 = (q2-9.55)/9.45
#             inte = 0
#             for j in range(len(c)):
#                 inte += P_l(j, q2) * c[j, 0, 0, i]
#             return inte
#         c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
#         
#         num += P_l(i, phi) * c_P_l
#     return num
# 
# acc = np.array([func4(i) for i in phi_range])
# 
# phi_bins = plt.hist((dataframe_without_res)["phi"], bins = 100)[0]
# plt.plot(phi_range, acc * phi_bins[0] / func4(-np.pi))
# plt.xlabel('phi')
# plt.ylabel('Counts')
# plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
# plt.show()
# plt.show()
# =============================================================================
