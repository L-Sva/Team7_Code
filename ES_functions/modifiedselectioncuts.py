#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:00:15 2022

@author: emirsezik
"""

import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import scipy.stats as ss
import pandas as pd
import glob as glob


dataframe = pd.read_pickle("Data/total_dataset.pkl")


def DLL(L_k, L_p):
    return np.log(L_k/L_p)


pickle_files = glob.glob("Data/*.pkl")
pickle_files.pop(0)

# %%
m_mu = 105.658
m_Pi = 139.57
m_K = 493.677
m_p = 938.27
mass_dict = {'mu_plus': m_mu,
             'mu_minus': m_mu,
             'Pi': m_Pi,
             'K': m_K,
             'p': m_p}


def cal_mass(dataframe, particles, mass):
    E = 0
    PX = 0
    PY = 0
    PZ = 0
    for i in range(len(particles)):
        E += np.sqrt(dataframe[particles[i] + '_P']
                     ** 2 + mass_dict[mass[i]] ** 2)
        PX += dataframe[particles[i] + '_PX']
        PY += dataframe[particles[i] + '_PY']
        PZ += dataframe[particles[i] + '_PZ']
    return np.sqrt(E**2 - PX**2 - PY**2 - PZ**2)

# %% List of all selection functions


def q2_resonances(data):
    q2 = data['q2']
    crit_a = (q2 > 8) & (q2 < 11)  # criteria A
    crit_b = (q2 > 12.5) & (q2 < 15)  # criteria B
    subset = data[~crit_a & ~crit_b]  # not crit_a and not crit_b
    not_subset = data[crit_a | crit_b]  # crit_a or crit_b
    return subset, not_subset


def q2_range(dataframe):
    q2 = dataframe['q2']
    crit = (q2 >= 0.1) & (q2 <= 19.0)
    subset = dataframe[crit]
    not_subset = dataframe[~crit]
    return subset, not_subset


def K0_vertex_chi2(dataframe):
    dataframe = dataframe.reset_index(drop=True)
    threshold = 8 * dataframe["Kstar_ENDVERTEX_NDOF"][0]
    subset = dataframe[dataframe['Kstar_ENDVERTEX_CHI2'] <= threshold]
    not_subset = dataframe[dataframe['Kstar_ENDVERTEX_CHI2'] > threshold]
    return subset, not_subset


def Kstar_inv_mass(dataframe):
    subset = []
    not_subset = []
    Kstar_MM = np.array(dataframe["Kstar_MM"])
    for i in range(len(Kstar_MM)):
        if (795.9 < Kstar_MM[i]) and (Kstar_MM[i] < 995.9):  # in MeV
            subset.append(i)
            continue
        else:
            not_subset.append(i)

    subset = dataframe.iloc[subset]

    not_subset = dataframe.iloc[not_subset]

    return subset, not_subset


def B0_vertex_chi2(dataframe):
    '''
    Input: dataframe - a dataframe need to be processed
            alpha - (in range 0 to 1) threshold of the propability of obtaining 
                    such a chi2; events with a large chi2 such that the probability
                    (of obtaining a chi2 as large or even larger) lower than alpha will be rejected.
    Output: subset - selected candidates
            not_subset - rejected candidates
    '''
    # def func(x):
    #     return ss.chi2.sf(x,5) - alpha    # scipy.chi2.sf() gives the survival function of Chi2 distribution, 5 degrees of freedom for vertex
    # threshold = float(fsolve(func, 5.))   #These 3 lines solves for the threshold chi2 value given the threshold alpha (probability)
    dataframe = dataframe.reset_index(drop=True)
    threshold = 8 * dataframe["B0_ENDVERTEX_NDOF"][0]
    subset = dataframe[dataframe['B0_ENDVERTEX_CHI2'] < threshold]
    not_subset = dataframe[dataframe['B0_ENDVERTEX_CHI2'] >= threshold]
    return subset, not_subset


def final_state_particle_IP(dataframe, threshold=9.):

    yes_1 = dataframe[dataframe['mu_plus_IPCHI2_OWNPV'] > threshold]
    no_1 = dataframe[dataframe['mu_plus_IPCHI2_OWNPV'] <= threshold]

    yes_2 = yes_1[yes_1['mu_minus_IPCHI2_OWNPV'] > threshold]
    no_2 = yes_1[yes_1['mu_minus_IPCHI2_OWNPV'] <= threshold]

    yes_3 = yes_2[yes_2['K_IPCHI2_OWNPV'] > threshold]
    no_3 = yes_2[yes_2['K_IPCHI2_OWNPV'] <= threshold]

    yes_4 = yes_3[yes_3['Pi_IPCHI2_OWNPV'] > threshold]
    no_4 = yes_3[yes_3['Pi_IPCHI2_OWNPV'] <= threshold]

    subset = yes_4

    not_subset = pd.concat([no_1, no_2, no_3, no_4])

    return subset, not_subset


def B0_IP_chi2(dataframe, threshold=16):
    accept = dataframe[dataframe['B0_IPCHI2_OWNPV'] < threshold]
    reject = dataframe[dataframe['B0_IPCHI2_OWNPV'] >= threshold]
    return accept, reject


def FD(dataframe, threshold=121):
    '''
    Input: dataframe - the dataframe need to be cleaned
            threshold - minimum flight distance accepted (default = 4.) in units of mm
    Output: subset and not_subset
    '''
    subset = dataframe[dataframe['B0_FDCHI2_OWNPV'] > threshold]
    not_subset = dataframe[dataframe['B0_FDCHI2_OWNPV'] <= threshold]
    return subset, not_subset


def KSTAR_FD(dataframe, threshold=16):
    '''
    Input: dataframe - the dataframe need to be cleaned
            threshold - minimum flight distance accepted (default = 4.) in units of mm
    Output: subset and not_subset
    '''
    subset = dataframe[dataframe['Kstar_FDCHI2_OWNPV'] > threshold]
    not_subset = dataframe[dataframe['Kstar_FDCHI2_OWNPV'] <= threshold]
    return subset, not_subset


def DIRA(dataframe, threshold=0.9999020016006562):
    subset = dataframe[dataframe['B0_DIRA_OWNPV'] > threshold]
    not_subset = dataframe[dataframe['B0_DIRA_OWNPV'] <= threshold]
    return subset, not_subset


def Particle_ID(dataframe):
    L_1 = ['mu_plus_MC15TuneV1_ProbNNmu', 'mu_minus_MC15TuneV1_ProbNNmu',
           'K_MC15TuneV1_ProbNNk', 'Pi_MC15TuneV1_ProbNNk']
    L_2 = ['mu_plus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNpi',
           'K_MC15TuneV1_ProbNNpi', 'Pi_MC15TuneV1_ProbNNpi']
    crit1 = (DLL(dataframe[L_1[0]], dataframe[L_2[0]]) > -3)
    crit2 = (DLL(dataframe[L_1[1]], dataframe[L_2[1]]) > -3)
    crit3 = (DLL(dataframe[L_1[2]], dataframe[L_2[2]]) > -5)
    crit4 = (DLL(dataframe[L_1[3]], dataframe[L_2[3]]) < 25)

    # subset = dataframe[ DLL(dataframe[ L_1[0] ], dataframe[ L_2[0] ] ) > -3]
    # subset = subset[ DLL(dataframe[ L_1[1] ], dataframe[ L_2[1] ] ) > -3]
    # subset = subset[ DLL(dataframe[ L_1[2] ], dataframe[ L_2[2] ] ) > -5]
    # subset = subset[ DLL(dataframe[ L_1[3] ], dataframe[ L_2[3] ] ) < 25]

    accept = crit1 & crit2 & crit3 & crit4
    reject = ~accept

    subset = dataframe[accept]
    not_subset = dataframe[reject]

    return subset, not_subset


def B0_mass(dataframe):

    crit = (dataframe["B0_MM"] > 5170) & (dataframe["B0_MM"] < 5700)

    subset = dataframe[crit]
    not_subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_1(dataframe):
    L_1 = 'Pi_MC15TuneV1_ProbNNk'
    L_2 = 'Pi_MC15TuneV1_ProbNNpi'

    crit = (DLL(dataframe[L_1], dataframe[L_2]) > -10) & \
        (cal_mass(dataframe, ['K', 'Pi'], ['K', 'K']) >= 1010) & (
            1030 >= cal_mass(dataframe, ['K', 'Pi'], ['K', 'K']))

    not_subset = dataframe[crit]
    subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_2(dataframe):
    L_1 = 'Pi_MC15TuneV1_ProbNNk'
    L_2 = 'Pi_MC15TuneV1_ProbNNpi'

    crit = (DLL(dataframe[L_1], dataframe[L_2]) > 10) & \
        (cal_mass(dataframe, ['K', 'Pi'], ['K', 'K']) >= 1030) & (
            1075 >= cal_mass(dataframe, ['K', 'Pi'], ['K', 'K']))

    not_subset = dataframe[crit]
    subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_3(dataframe):
    crit = (cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], ['K', 'K', 'mu_plus', 'mu_minus']) >= 5321) & (
        5411 >= cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], ['K', 'K', 'mu_plus', 'mu_minus']))

    not_subset = dataframe[crit]
    subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_4(dataframe):
    L_1 = ['K_MC15TuneV1_ProbNNk', 'Pi_MC15TuneV1_ProbNNk']
    L_2 = ['K_MC15TuneV1_ProbNNpi', 'Pi_MC15TuneV1_ProbNNpi']

    crit = (DLL(dataframe[L_1[0]], dataframe[L_2[0]]) >
            DLL(dataframe[L_1[1]], dataframe[L_2[1]]))

    subset = dataframe[crit]
    not_subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_5(dataframe):
    L_1 = 'Pi_MC15TuneV1_ProbNNp'
    L_2 = 'Pi_MC15TuneV1_ProbNNpi'

    crit = (DLL(dataframe[L_1], dataframe[L_2]) > 0) & (5665 >= cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], [
        'K', 'p', 'mu_plus', 'mu_minus'])) & (cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], ['K', 'p', 'mu_plus', 'mu_minus']) >= 5575)

    not_subset = dataframe[crit]
    subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_6(dataframe):
    L_1 = 'Pi_MC15TuneV1_ProbNNk'
    L_2 = 'Pi_MC15TuneV1_ProbNNpi'

    crit = (DLL(dataframe[L_1], dataframe[L_2]) > 0) & (5665 >= cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], [
        'p', 'K', 'mu_plus', 'mu_minus'])) & (cal_mass(dataframe, ['K', 'Pi', 'mu_plus', 'mu_minus'], ['p', 'K', 'mu_plus', 'mu_minus']) >= 5575)

    not_subset = dataframe[crit]
    subset = dataframe[~crit]

    return subset, not_subset


def peaking_back_7(dataframe):
    L_1 = 'Pi_MC15TuneV1_ProbNNmu'
    L_2 = 'Pi_MC15TuneV1_ProbNNpi'

    crit1 = (DLL(dataframe[L_1], dataframe[L_2]) > 5.0) & (3156 >= cal_mass(dataframe, ['Pi', 'mu_plus'], [
        'mu_minus', 'mu_plus'])) & (cal_mass(dataframe, ['Pi', 'mu_plus'], ['mu_minus', 'mu_plus']) >= 3036)
    crit2 = (DLL(dataframe[L_1], dataframe[L_2]) > 5.0) & (3156 >= cal_mass(dataframe, ['Pi', 'mu_minus'], [
        'mu_plus', 'mu_minus'])) & (cal_mass(dataframe, ['Pi', 'mu_minus'], ['mu_plus', 'mu_minus']) >= 3036)

    not_subset = dataframe[crit1 | crit2]
    subset = dataframe[~crit1 & ~crit2]

    return subset, not_subset


def peaking_back_8(dataframe):
    L_1 = 'K_MC15TuneV1_ProbNNmu'
    L_2 = 'K_MC15TuneV1_ProbNNk'

    crit1 = (DLL(dataframe[L_1], dataframe[L_2]) > 5.0) & (3156 >= cal_mass(dataframe, ['K', 'mu_plus'], [
        'mu_minus', 'mu_plus'])) & (cal_mass(dataframe, ['K', 'mu_plus'], ['mu_minus', 'mu_plus']) >= 3036)
    crit2 = (DLL(dataframe[L_1], dataframe[L_2]) > 5.0) & (3156 >= cal_mass(dataframe, ['K', 'mu_minus'], [
        'mu_plus', 'mu_minus'])) & (cal_mass(dataframe, ['K', 'mu_minus'], ['mu_plus', 'mu_minus']) >= 3036)

    not_subset = dataframe[crit1 | crit2]
    subset = dataframe[~crit1 & ~crit2]

    return subset, not_subset


# def Particle_ID(dataframe_1):

#     dataframe = dataframe_1.copy()

#     # Example of how one could go about vectorising this
#     n1 = dataframe['mu_plus_MC15TuneV1_ProbNNk'].to_numpy()
#     n2 = dataframe['mu_plus_MC15TuneV1_ProbNNpi'].to_numpy()
#     n3 = dataframe['mu_plus_MC15TuneV1_ProbNNmu'].to_numpy()
#     n4 = dataframe['mu_plus_MC15TuneV1_ProbNNe'].to_numpy()
#     n5 = dataframe['mu_plus_MC15TuneV1_ProbNNp'].to_numpy()
#     crit_1 = (n3 > n1) & (n3 > n2) & (n3 > n4) & (n3 > n5)

#     n1 = dataframe['mu_minus_MC15TuneV1_ProbNNk'].to_numpy()
#     n2 = dataframe['mu_minus_MC15TuneV1_ProbNNpi'].to_numpy()
#     n3 = dataframe['mu_minus_MC15TuneV1_ProbNNmu'].to_numpy()
#     n4 = dataframe['mu_minus_MC15TuneV1_ProbNNe'].to_numpy()
#     n5 = dataframe['mu_minus_MC15TuneV1_ProbNNp'].to_numpy()
#     crit_2 = (n3 > n1) & (n3 > n2) & (n3 > n4) & (n3 > n5)

#     n1 = dataframe['K_MC15TuneV1_ProbNNk'].to_numpy()
#     n2 = dataframe['K_MC15TuneV1_ProbNNpi'].to_numpy()
#     n3 = dataframe['K_MC15TuneV1_ProbNNmu'].to_numpy()
#     n4 = dataframe['K_MC15TuneV1_ProbNNe'].to_numpy()
#     n5 = dataframe['K_MC15TuneV1_ProbNNp'].to_numpy()
#     crit_3 = (n1 > n2) & (n1 > n3) & (n1 > n4) & (n1 > n5)

#     n1 = dataframe['Pi_MC15TuneV1_ProbNNk'].to_numpy()
#     n2 = dataframe['Pi_MC15TuneV1_ProbNNpi'].to_numpy()
#     n3 = dataframe['Pi_MC15TuneV1_ProbNNmu'].to_numpy()
#     n4 = dataframe['Pi_MC15TuneV1_ProbNNe'].to_numpy()
#     n5 = dataframe['Pi_MC15TuneV1_ProbNNp'].to_numpy()
#     crit_4 = (n2 > n1) & (n2 > n3) & (n2 > n4) & (n2 > n5)

#     accept = crit_1 & crit_2 & crit_3 & crit_4
#     reject = ~accept

#     subset = dataframe[accept]
#     not_subset = dataframe[reject]

#     return subset, not_subset
# %% Selection Criteria ALL
def selection_pb(dataframe):
    yes_pb1, no_pb1 = peaking_back_1(dataframe)

    yes_pb2, no_pb2 = peaking_back_2(yes_pb1)

    yes_pb3, no_pb3 = peaking_back_3(yes_pb2)

    yes_pb4, no_pb4 = peaking_back_4(yes_pb3)

    yes_pb5, no_pb5 = peaking_back_5(yes_pb4)

    yes_pb6, no_pb6 = peaking_back_6(yes_pb5)

    yes_pb7, no_pb7 = peaking_back_7(yes_pb6)

    yes_pb8, no_pb8 = peaking_back_8(yes_pb7)

    return yes_pb8, pd.concat([no_pb1, no_pb2, no_pb3, no_pb4, no_pb5, no_pb6, no_pb7, no_pb8])


def selection_all(dataframe,
                  final_particle_prob_threshold=9., B0_IP_chi2_threshold=16,
                  B0_FD_threshold=121, DIRA_threshold=0.9999, ):
    """ Manual selectors including the q2 resonances selector """

    yes_q2, no_q2 = q2_resonances(dataframe)

    yes_other, no_other = selection_all_withoutres(
        dataframe, final_particle_prob_threshold, B0_IP_chi2_threshold, B0_FD_threshold, DIRA_threshold)

    no = [no_q2, no_other]
    not_subset = pd.concat(no)
    subset = yes_other

    return subset, not_subset

# %%


def selection_all_withoutres(dataframe,
                             final_particle_prob_threshold=9., B0_IP_chi2_threshold=16,
                             B0_FD_threshold=121, DIRA_threshold=0.9999, ):
    """ Manual selectors excluding the q2_resonances selector """

    yes_PID, no_PID = Particle_ID(dataframe)

    yes_q2range, no_q2range = q2_range(yes_PID)

    yes_Kstar_mass, no_Kstar_mass = Kstar_inv_mass(yes_q2range)

    yes_B0_vertex, no_B0_vertex = B0_vertex_chi2(yes_Kstar_mass)

    yes_B0_IP, no_B0_IP = B0_IP_chi2(yes_B0_vertex, B0_IP_chi2_threshold)

    yes_fs_IP, no_fs_IP = final_state_particle_IP(
        yes_B0_IP, final_particle_prob_threshold)

    yes_FD, no_FD = FD(yes_fs_IP, B0_FD_threshold)

    yes_DIRA, no_DIRA = DIRA(yes_FD, DIRA_threshold)

    yes_KOV, no_KOV = K0_vertex_chi2(yes_DIRA)

    yes_KFD, no_KFD = KSTAR_FD(yes_KOV)

    yes_b0_mass, no_b0_mass = B0_mass(yes_KFD)

    yes_pball, no_pball = selection_pb(yes_b0_mass)

    no = [no_q2range, no_Kstar_mass, no_B0_vertex, no_B0_IP, no_fs_IP, no_FD,
          no_DIRA, no_KOV, no_KFD, no_b0_mass, no_pball]
    not_subset = pd.concat(no)
    subset = yes_pball

    return subset, not_subset
# %% Test

# L_1 = ['mu_plus_MC15TuneV1_ProbNNmu', 'mu_minus_MC15TuneV1_ProbNNmu',
#        'Pi_MC15TuneV1_ProbNNk', 'K_MC15TuneV1_ProbNNk']
# L_2 = ['mu_plus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNpi',
#        'K_MC15TuneV1_ProbNNpi', 'Pi_MC15TuneV1_ProbNNpi']
# %%
# =============================================================================
# if __name__ == "__main__":
#     selected, not_selected = selection_all(dataframe)
#     print(len(selected), len(not_selected))
#
# #%%
#
#
# no_init = []
# no_final = []
#
# for i in range(len(pickle_files)):
#     data = pd.read_pickle(pickle_files[i])
#     no_init.append(  len(data) )
#     selected, not_selected = selection_all(data)
#     no_final.append(len(selected))
#
#
# for i in range(len(pickle_files)):
#     print("In %s, there were %g events of which %g survived" %
#       (pickle_files[i], no_init[i], no_final[i]) )
#
# =============================================================================
