"""
@author: Sam Zhou
"""
# starting with the dataframe 'selected', which needs to be defined by filtering for the candidates of a give data file

# binning over q2
q2bins = [[0.1, 0.98],
          [1.1, 2.5],
          [2.5, 4.0],
          [4.0, 6.0],
          [6.0, 8.0],
          [15.0, 17.0],
          [17.0, 19.0],
          [11.0, 12.5],
          [1.0, 6.0],
          [15.0, 17.9]]
def q2binning(data, q2bins):
    binned_list =[]
    for i in range(len(q2bins)):
        q2 = data['q2']
        subset = (q2 > q2bins[i][0]) & (q2 <= q2bins[i][1])
        binned_list.append(data[subset]) 
    return binned_list

q2binned_data = q2binning(selected, q2bins)

# in each q2 binning over cos theta_l
ctheta_l_bins = [[-1, -0.5],
          [-0.5, 0],
          [0, 0.5],
          [0.5, 1]]
def ctheta_l_binning(data, ctheta_l_bins):
    binned_list =[]
    for i in range(len(ctheta_l_bins)):
        costhetal = data['costhetal']
        subset = (costhetal > ctheta_l_bins[i][0]) & (costhetal <= ctheta_l_bins[i][1])
        binned_list.append(data[subset]) 
    return binned_list

# repeated binning for all q2 bins
binned = []
for i in range(len(q2binned_data)):
    binned_costhetal = ctheta_l_binning(q2binned_data[i], ctheta_l_bins)
    binned.append(binned_costhetal)

# The final result is in the list called 'binned'. Within it, each is a list corresponding to a q2 bin. Within each q2 bin, each is a list corresponding to a costhetal bin.
