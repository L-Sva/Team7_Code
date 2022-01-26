"""
this code takes the cosine of the angle between the B_0 flight vector and momentum vector with respect to the primary vertex (B0_DIRA_OWNPV) from the total dataset 
and appends the serial number of data above a threshold value to a candidate list
a B_0 candidate should have the momentum vector in line with the flight vector, so the cosine of the angle must be close to 1
"""
import pandas as pd

all_data = pd.read_csv('total_dataset.csv') 

all_data = all_data.to_numpy()
B0_DIRA_OWNPV = all_data[:,[70]].flatten()
index = all_data[:,[0]].flatten()

threshold = 0.9999999 

candidates = [] 

for i in range(len(B0_DIRA_OWNPV)):
    if B0_DIRA_OWNPV[i] > threshold:
        candidates.append(index[i])
    else:
        continue
