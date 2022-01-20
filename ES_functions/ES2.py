import pandas as pd
import numpy as np
#%%
"""
Function
"""
def ES2(dataframe):
    subset = []
    not_subset = []
    Kstar_MM = np.array(dataframe["Kstar_MM"])
    for i in range(len(Kstar_MM)):
        if (795.9 < Kstar_MM[i]) and (Kstar_MM[i] < 995.9): #in MeV
            subset.append(i)
            continue
        else:
            not_subset.append(i)

    subset = dataframe.iloc[subset]

    not_subset = dataframe.iloc[not_subset]

    return subset, not_subset
#%%
#test
file=open('total_dataset.csv')
dataset=pd.read_csv('total_dataset.csv',delimiter=',')
subset, not_subset=ES2(dataset)
print(subset["Kstar_MM"].head())
