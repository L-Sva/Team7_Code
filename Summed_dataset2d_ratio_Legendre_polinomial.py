import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import load_file
from ES_functions.Compiled import selection_all

plt.rcParams['font.size'] = 18

def load_q2_binned():
    all_signal_files = [
        'jpsi_mu_k_swap', 
        'jpsi_mu_pi_swap', 
        'jpsi', 
        'k_pi_swap', 
        'phimumu',
        'pKmumu_piTok_kTop', 
        'pKmumu_piTop', 
        'psi2S', 
        'signal'
    ]

    data_dfs = [load_file(f'{file_name}.pkl')
                for file_name in all_signal_files]
    
    total_df = pd.concat(data_dfs) # all signal + peaking in one dataframe
    selected_df, _ = selection_all(total_df)
    raw_signal_df = load_file('signal.pkl')
    
    # bins ordered different, because np.histogram needs a
    # monotonically increasing array
    q2_bins_0 = [0.1, 0.98, 1.1, 2.5, 4, 6, 8, 15, 17, 19]
    q2_bins_1 = [1, 6, 11, 12.5, 15, 17.9]

    rs_binned_0, bin_edges_0 = pd.cut(
        raw_signal_df['q2'], bins=q2_bins_0, retbins=True,
        labels=['0', 'i1', '1', '2', '3', '4', 'i2', '5', '6']
    )
    rs_binned_1, bin_edges_1 = pd.cut(
        raw_signal_df['q2'], bins=q2_bins_1, retbins=True,
        labels=['7', 'i3', '8', 'i4', '9']
    )
    selected_binned_0 = pd.cut(
        selected_df['q2'], bins=q2_bins_0,
        labels=['0', 'i1', '1', '2', '3', '4', 'i2', '5', '6']
    )
    selected_binned_1 = pd.cut(
        selected_df['q2'], bins=q2_bins_1,
        labels=['7', 'i3', '8', 'i4', '9']
    )

    rs_dfs_0 = dict(tuple(raw_signal_df.groupby(rs_binned_0)))
    rs_dfs_1 = dict(tuple(raw_signal_df.groupby(rs_binned_1)))

    selected_dfs_0 = dict(tuple(selected_df.groupby(selected_binned_0)))
    selected_dfs_1 = dict(tuple(selected_df.groupby(selected_binned_1)))

    raw_signal_dfs = {**rs_dfs_0, **rs_dfs_1}
    selected_dfs = {**selected_dfs_0, **selected_dfs_1}

    return raw_signal_dfs, selected_dfs
#%%
raw_signal_dfs, selected_dfs = load_q2_binned()

# save the above 2 variables to avoid re-calculating and faster loading
import pickle
with open('ratio_data.pkl', 'wb') as ratio_file:
    pickle.dump([raw_signal_dfs, selected_dfs], ratio_file)
#%%
# once saved, comment out lines 63, 67 and 68

def get_ratio(q2bin,cosbin_num):
    
    ratio=[] # Acceptance ratio F(q2,costhetal) 2d array
    with open('ratio_data.pkl', 'rb') as ratio_file:
        raw_signal_dfs, selected_dfs = pickle.load(ratio_file)

    for i in range (len(q2bin)):
        
        # an example code:
        bin_no = q2bin[i] # change me to be your assigned bin
    
        rs_q0 = raw_signal_dfs[f'{bin_no}']
        selected_q0 = selected_dfs[f'{bin_no}']
    
        counts_rs, bin_edges = np.histogram(rs_q0['costhetal'], bins=cosbin_num)
        counts_selected, _ = np.histogram(selected_q0['costhetal'], bins=bin_edges)
    
        ratio_inq2bin = counts_selected/counts_rs
        ratio.append(ratio_inq2bin)
    
    return ratio

#%% Define Legendre polinomial
def P_0(x):
    return 1
def P_1(x):
    return x
def P_2(x):
    return 0.5*(3*x**2-1)
def P_3(x):
    return 0.5*(5*x**3-3*x)
def P_4(x):
    return 0.125*(35*x**4-30*x**2+3)
def P_5(x):
    return 0.125*(63*x**5-70*x**3+15*x)

P=[P_0,P_1,P_2,P_3,P_4,P_5] #Legendre polinomial to 5th order
    

#%% Using legendre polynomial to construct acceptance function F(q2,costhetal)

# q2bins=['0', 'i1', '1', '2', '3', '4', 'i2', '5', '6'] #
q2bins=['0', '1', '2', '3', '4', '5', '6']

n_cosbin=100 #number of costhetal bins

ratio=get_ratio(q2bins,n_cosbin) # fitting points of F(q2,costhetal)

q2_value=np.array([0.54, 1.8, 3.25, 5, 7, 16, 18]) #using the midpoint 

q2_normal=(1/9.45)*(q2_value-9.55) #nomalization q2_normal has range [-1,1]

q2_bw=np.array([0.88, 1.4, 1.5, 2, 2, 2, 2]) #q2bin width

cos_bw=2/n_cosbin # bin width of costhetal

costhetal=np.linspace(-1+2/n_cosbin,1-2/n_cosbin,n_cosbin) # midpoint of costhetal bin

c=np.zeros([10,10])# coefficients of the 2d legendre polynomial c_ij

for i in range (0,6): 
    for j in range (0,6): 
        SUM1=0
        for k in range (len(q2_normal)): # sum up all q2 values
            SUM2=0
            for s in range (n_cosbin): # sum up all costhetal values
                SUM2+=cos_bw*P[i](q2_normal[k])*P[j](costhetal[s])*ratio[k][s]
            SUM1+=q2_bw[k]*SUM2
        c[i][j]=(2*i+1)/2*(2*j+1)/2*SUM1
#%%
np.savetxt('acceptance_func_parameters/costhetal_0.txt', costhetal)
np.savetxt('acceptance_func_parameters/q2_normal_0.txt', q2_normal)
np.savetxt('acceptance_func_parameters/c_0.txt', c)
#%% Plot the acceptance function of the first q2 bin (i.e. i=0)
costhetal=np.linspace(-0.99,0.99,1000)
def F (cos,q2,coe): # Continuous 2d acceptance function F(q2_normal,costhetal). Here we let q2_normal be fixed
    SUM=0
    for i in range (0,6): 
        for j in range (0,6): 
            SUM+=c[i][j]*P[i](q2)*P[j](cos)
    return SUM

plt.plot(costhetal,F(costhetal,q2_normal[0],c))
    
 


#%%
# an example code:
bin_no = '0' # change me to be your assigned bin

rs_q0 = raw_signal_dfs[f'{bin_no}']
selected_q0 = selected_dfs[f'{bin_no}']

counts_rs, bin_edges = np.histogram(rs_q0['costhetal'], bins=100)
counts_selected, _ = np.histogram(selected_q0['costhetal'], bins=bin_edges)

ratio = counts_selected/counts_rs

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(bin_edges[:-1], ratio, 'k.')
# ax.bar(
#     bin_edges[:-1], ratio, width=np.diff(bin_edges),
#    edgecolor='black', align='edge'
# )
ax.set(
    xlabel=r'$cos(\theta_l)$', ylabel='Accepted ratio', title=f'Bin {bin_no}'
)
plt.show()

#%%

q2bins_1=['7', '8', '9']

n_cosbin=100 #number of costhetal bins

ratio_1=get_ratio(q2bins_1,n_cosbin) # fitting points of F(q2,costhetal)

q2_value_1=np.array([3.5, 11.75, 16.45]) #using the midpoint 

q2_normal_1=(1/9.45)*(q2_value_1-9.55) #nomalization q2_normal has range [-1,1]

q2_bw_1=np.array([5, 1.5, 2.9]) #q2bin width

cos_bw=2/n_cosbin # bin width of costhetal

costhetal=np.linspace(-1+2/n_cosbin,1-2/n_cosbin,n_cosbin) # midpoint of costhetal bin

c_1=np.zeros([10,10])# coefficients of the 2d legendre polynomial c_ij

for i in range (0,6): 
    for j in range (0,6): 
        SUM1=0
        for k in range (len(q2_normal_1)): # sum up all q2 values
            SUM2=0
            for s in range (n_cosbin): # sum up all costhetal values
                SUM2+=cos_bw*P[i](q2_normal_1[k])*P[j](costhetal[s])*ratio_1[k][s]
            SUM1+=q2_bw[k]*SUM2
        c_1[i][j]=(2*i+1)/2*(2*j+1)/2*SUM1

#%% Plot the acceptance function of the first q2 bin (i.e. i=0)
costhetal_1=np.linspace(-0.99,0.99,1000)
def F (cos,q2,coe): # Continuous 2d acceptance function F(q2_normal,costhetal). Here we let q2_normal be fixed
    SUM=0
    for i in range (0,6): 
        for j in range (0,6): 
            SUM+=c[i][j]*P[i](q2)*P[j](cos)
    return SUM

plt.plot(costhetal_1,F(costhetal_1,q2_normal_1[0],c_1))

np.savetxt('acceptance_func_parameters/costhetal_1.txt', costhetal_1)
np.savetxt('acceptance_func_parameters/q2_normal_1.txt', q2_normal_1)
np.savetxt('acceptance_func_parameters/c_1.txt', c_1)