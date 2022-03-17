#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:44:03 2022

@author: zwl0331
"""
from find_acceptance_new import *
import matplotlib.pyplot as plt
from scipy.integrate import quad

#%% check q2
q2_n, q2_bins, _ = plt.hist(dataframe_without_res["q2"], bins = 1000)
 
q2_range = np.linspace(0.1, 19, 1000)
def func(q2):
     q2 = (q2-9.55)/9.45
     num = 0
     for i in range(len(c)):
         num += P_l(i, q2) * c[i, 0, 0, 0]
     return num
acc = [func(i) for i in q2_range]
plt.plot(q2_range, np.array(acc) * q2_n[0] / func(0.1))
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
ctl_range = np.linspace(-1, 1, 100)
def func2(ctl):
    num = 0
    for i in range(len(c[0])):
        
        c_P_l= 0
        def inte(q2):
            q2 = (q2-9.55)/9.45
            inte = 0
            for j in range(len(c)):
                inte += P_l(j, q2) * c[j, i, 0, 0]
            return inte
        c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
        
        num += P_l(i, ctl) * c_P_l
    return num
acc = np.array([func2(i) for i in ctl_range])

ctl_n, ctl_bins, _ = plt.hist(dataframe_without_res["costhetal"], bins = 100)
plt.plot(ctl_range, acc * ctl_n[0] / func2(-1)*0.95)
plt.xlabel('ctl')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()
plt.show()

#%% check ctk
ctk_range = np.linspace(-1, 1, 100)
def func3(ctk):
    num = 0
    for i in range(len(c[0, 0])):
        
        c_P_l= 0
        def inte(q2):
            q2 = (q2-9.55)/9.45
            inte = 0
            for j in range(len(c)):
                inte += P_l(j, q2) * c[j, 0, i, 0]
            return inte
        c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
        
        num += P_l(i, ctk) * c_P_l
    return num
acc = np.array([func3(i) for i in ctk_range])

ctk_n, ctk_bins, _ = plt.hist((dataframe_without_res)["costhetak"], bins = 100)
plt.plot(ctk_range, acc * ctk_n[0] / func3(-1))
plt.xlabel('ctk')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()

#%% check phi
phi_range = np.linspace(-np.pi, np.pi, 100)
def func4(phi):
    num = 0
    phi = phi/np.pi
    for i in range(len(c[0, 0, 0])):
        
        c_P_l= 0
        def inte(q2):
            q2 = (q2-9.55)/9.45
            inte = 0
            for j in range(len(c)):
                inte += P_l(j, q2) * c[j, 0, 0, i]
            return inte
        c_P_l = quad(inte, 0.1, 8)[0] + quad(inte, 11, 12.5)[0] + quad(inte, 15, 19)[0]
        
        num += P_l(i, phi) * c_P_l
    return num

acc = np.array([func4(i) for i in phi_range])

phi_n, phi_bins, _ = plt.hist((dataframe_without_res)["phi"], bins = 100)
plt.plot(phi_range, acc * phi_n[0] / func4(-np.pi))
plt.xlabel('phi')
plt.ylabel('Counts')
plt.legend(['Acceptance Function', 'Filtered acceptance_mc'])
plt.show()
plt.show()