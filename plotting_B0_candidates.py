import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from core import load_file, RAWFILES
from ES_functions.Compiled import selection_all
from fitting.functions import q2_binned, q2bins
from histrogram_plots import plot_hist_quantity
#%%
def fit0(x,A2,mu1,std1):
    gaussian1 = A2*np.exp(-(x-mu1)**2/(2*std1**2))
    return gaussian1

def twogaussian(x,A2,mu1,std1,A3,std2):
    gaussian1 = A2*np.exp(-(x-mu1)**2/(2*std1**2))
    gaussian2 = A3*np.exp(-(x-mu1)**2/(2*std2**2))
    return gaussian1+gaussian2

def fit1(x,A2,mu1,std1,A1,l):
    exponential = A1*np.exp(-x/l)
    gaussian1 = A2*np.exp(-(x-mu1)**2/(2*std1**2))
    f = exponential+gaussian1
    return f

def fit2(x,A2,mu1,std1,A1,l,A3,std2):
    exponential = A1*np.exp(-x/l)
    gaussian1 = A2*np.exp(-(x-mu1)**2/(2*std1**2))
    gaussian2 = A3*np.exp(-(x-mu1)**2/(2*std2**2))
    f = exponential+gaussian1+gaussian2
    return f

def exponential(x,A1,l):
    exponential = A1*np.exp(-x/l)
    return exponential

#%%
raw_total = load_file(RAWFILES.TOTAL_DATASET)
filtered_total, _ = selection_all(raw_total)

q2_filtered = q2_binned(filtered_total)
#%%
bin_no=40
bins=[]
h=[]
for i in range(10):
    plt.figure(i)
    bin1, _, h1 = plot_hist_quantity(q2_filtered[f'{i}'],column='B0_MM',bins=bin_no)
    bins.append(bin1)
    h.append(h1)
    plt.close()

bin_mids=[]
for j in range(10):
    bin_mid=[(bins[j][i]+bins[j][i+1])/2 for i in range(len(bins[j])-1)]
    bin_mids.append(bin_mid)
bin_mids=np.array(bin_mids)
#%%

for num in range(10):
    p00=[2e-2,5275,20]
    bounds0=[[0,-np.inf,0],[np.inf,np.inf,np.inf]]
    vals0, cov0 = opt.curve_fit(fit0,bin_mids[num],h[num],p0=p00,bounds=bounds0)
    
    vals0=np.ndarray.tolist(vals0)
    # plt.figure(0)
    # x=np.linspace(bins[num][0],bins[num][-1],1000)
    # y=fit0(x,*vals0)
    # plt.plot(x,y,color='black')
    # plot_hist_quantity(q2_filtered[f'{num}'],column='B0_MM',bins=bin_no)
    # plt.show()
    # print(np.round(vals0,decimals=2))
    
    p01=vals0+[2e5,2e2]
    bounds1=[[0,-np.inf,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
    vals1, cov1 = opt.curve_fit(fit1,bin_mids[num],h[num],p0=p01,bounds=bounds1)
    
    
    vals1=np.ndarray.tolist(vals1)
    
    
    # plt.figure(1)
    # x=np.linspace(bins[num][0],bins[num][-1],1000)
    # y=fit1(x,*vals1)
    # plt.plot(x,y,color='black')
    # plot_hist_quantity(q2_filtered[f'{num}'],column='B0_MM',bins=bin_no)
    # plt.show()
    # print(np.round(vals1,decimals=2))
    
    # p03=vals0+[vals0[0]]+[vals0[2]]
    # bounds3=[[0,-np.inf,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
    # vals3, cov3 = opt.curve_fit(twogaussian,bin_mids[num],h[num],p0=p03,bounds=bounds3)
    
    # plt.figure(1)
    # x=np.linspace(bins[num][0],bins[num][-1],1000)
    # y=twogaussian(x,*vals3)
    # plt.plot(x,y,color='black')
    # plot_hist_quantity(q2_filtered[f'{num}'],column='B0_MM',bins=bin_no)
    # plt.show()
    # vals3=np.ndarray.tolist(vals3)
    
    
    p02=vals1+[vals1[0]]+[vals1[2]]
    bounds2=[[0,-np.inf,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]]
    
    # p02=vals3[:2]+vals1[2:]+vals3[3:]
    vals2, cov2 = opt.curve_fit(fit2,bin_mids[num],h[num],p0=p02,bounds=bounds2)
    
    x=np.linspace(bins[num][0],bins[num][-1],1000)
    y=fit2(x,*vals2)
    yexp=exponential(x,vals2[3],vals2[4])
    yzero=np.zeros(len(yexp))
    plt.figure(num)
    plt.plot(x,y,color='black')
    plt.fill_between(x,yzero,yexp)
    plot_hist_quantity(q2_filtered[f'{num}'],column='B0_MM',bins=bin_no)
    plt.title(f'{q2bins[num][0]} < q^2 < {q2bins[num][1]}')
    plt.xlabel('B0_MM (MeV/c^2)')
    plt.show()
    plt.savefig(f'B0_graphs/{q2bins[num][0]}-{q2bins[num][1]}.png', dpi=500)
    plt.close()
    print(np.round(vals2,decimals=2))
