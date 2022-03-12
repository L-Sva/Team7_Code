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

from modifiedselectioncuts import selection_all_withoutres, q2_resonances
from ml_selector import remove_combinatorial_background

plt.rcParams['font.size'] = 18

Path('../tmp/').mkdir(exist_ok=True)

#%%
def P_l(n, x):
    
    n = np.asarray(n)
    scalar_input_n = False
    if n.ndim == 0:
        n = n[None]  # Makes x 1D
        scalar_input_n = True
    
    x = np.asarray(x)
    scalar_input_x = False
    if x.ndim == 0:
        x = x[None]  # Makes x 1D
        scalar_input_x = True
        
    coeff = np.array([legendre(i) for i in n], dtype=object)
    ret = np.zeros([len(coeff), len(x)])
    for i in range(len(coeff)):
        for j in range(len(x)):
            ret[i, j] = np.poly1d(np.array(coeff[i]))(x[j])
            
    axis = []
    if scalar_input_n:
        axis.append(0)
    if scalar_input_x:
        axis.append(1)
    axis = tuple(axis)

    return np.squeeze(ret, axis = axis)

# =============================================================================
# #%%
# def acceptance_function(q2, ctl, ctk, phi, coeff):
#     q2 = (q2-9.55)/9.45
#     phi = phi/np.pi
#     acc_func = 0
#     i_max, j_max, k_max, l_max = coeff.shape
#     for i in range(i_max):
#         for j in range(j_max):
#             for k in range(k_max):
#                 for l in range(l_max):
#                     acc_func += coeff[i,j,k,l] * P_l(i, q2) * P_l(j, ctl) * P_l(k, ctk) * P_l(l, phi)
#     return acc_func
# 
# 
# =============================================================================

def acceptance_function(q2, ctl, ctk, phi, coeff):
    q2 = np.asarray(q2)
    ctl = np.asarray(ctl)
    ctk = np.asarray(ctk)
    phi = np.asarray(phi)
    scalar_input = False
    if q2.ndim == 0:
        q2 = q2[None]  # Makes x 1D
        ctl = ctl[None]  # Makes x 1D
        ctk = ctk[None]  # Makes x 1D
        phi = phi[None]  # Makes x 1D
        scalar_input = True
    res = np.zeros(q2.shape, dtype = object)
    
    q2 = (q2-9.55)/9.45
    phi = phi/np.pi
    i_max, j_max, k_max, l_max = coeff.shape
    for i in range(len(q2)):
    
        inter1 = np.tensordot(P_l(np.arange(i_max), q2[i]), P_l(np.arange(j_max), ctl[i]), axes = 0)
        inter2 = np.tensordot(inter1, P_l(np.arange(k_max), ctk[i]), axes = 0)
        inter3 = np.tensordot(inter2, P_l(np.arange(l_max), phi[i]), axes = 0)
        inter4 = np.tensordot(coeff, inter3, axes = ((0, 1, 2, 3), (0, 1, 2, 3)))
        res[i] = inter4
    if scalar_input:
        res = np.squeeze(res, axis = 0)
    return res

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

#%% run this section if this is the first time computing acceptance function
# filter acceptance_mc
dataframe = pd.read_pickle("data/acceptance_mc.pkl")
dataframe, _ = selection_all_withoutres(dataframe)
dataframe_with_res, _ = remove_combinatorial_background(dataframe)
dataframe_with_res.to_pickle('../tmp/filtered_acc_with_res.pkl')
dataframe_without_res, _ = q2_resonances(dataframe_with_res)
dataframe_without_res.to_pickle('../tmp/filtered_acc_without_res.pkl')
c = coeff(dataframe_with_res)
np.save('../tmp/coeff.npy', c)

#%% run this section to avoid repeated calculation
dataframe_with_res = pd.read_pickle('../tmp/filtered_acc_with_res.pkl')
dataframe_without_res = pd.read_pickle('../tmp/filtered_acc_without_res.pkl')
c = np.load('../tmp/coeff.npy')

# =============================================================================
# def func(q2, ctl):
#     acc = 0
#     q2 = (q2 -9.55) / 9.45
#     for i in range(len(c)):
#         for j in range(len(c[0])):
#             acc += P_l(i, q2) * P_l(j, ctl) * c[i, j, 0, 0]
#             
#     return acc
# q2 = np.linspace(0.1, 19, 100)
# ctl = np.linspace(-1, 1, 100)
# q2m, ctlm = np.meshgrid(q2, ctl)
# scalar = func(q2m, ctlm)
# plt.imshow(scalar)
# plt.colorbar()
# =============================================================================

# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(ctlm, q2m, scalar)
# ax.plot_wireframe(ctlm, q2m, np.zeros(ctlm.shape))
# =============================================================================

#%%


# =============================================================================
# #All the following are testing
# 
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

