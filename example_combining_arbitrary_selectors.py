from core import B0_MM_selector, load_file, RAWFILES, combine_n_selectors
from ES_functions.Compiled import (q2_resonances, Kstar_inv_mass, B0_vertex_chi2,
 final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID)
from histrogram_plots import generic_selector_plot
import itertools
import numpy as np
import matplotlib.pyplot as plt

both = combine_n_selectors(q2_resonances, B0_vertex_chi2, B0_MM_selector)

total_dataset = load_file(RAWFILES.TOTAL_DATASET)
s, ns = both(total_dataset, B0_vertex_chi2 = 0.2)

print(len(s),len(ns))

#generic_selector_plot(total_dataset, s, ns, 'q2')
#generic_selector_plot(total_dataset, s, ns, 'B0_ENDVERTEX_CHI2')
#generic_selector_plot(total_dataset, s, ns, 'B0_MM')

selectors = (q2_resonances, Kstar_inv_mass, B0_vertex_chi2,
 final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID)

params = {
    'B0_vertex_chi2': 0.2,
    'final_state_particle_IP': 0.02, 
    'B0_IP_chi2': 9, 
    'FD': 10, 
    'DIRA': 0.9999999,
}

pairs = list(itertools.combinations(selectors, 3))
remaining_events = [
    len(combine_n_selectors(*pair)(total_dataset, **params)[0]) for pair in pairs
]

ix = np.argsort(remaining_events)

pairs = np.array(pairs)[ix]
remaining_events = np.array(remaining_events)[ix]

print('Most restrictive selector combinations')
for i in range(10):
    print([selector.__name__ for selector in pairs[i]], remaining_events[i])

# Output for pairs
# ['q2_resonances', 'DIRA'] 2284
# ['q2_resonances', 'Particle_ID'] 5889
# ['q2_resonances', 'B0_vertex_chi2'] 13989
# ['q2_resonances', 'FD'] 15738
# ['DIRA', 'Particle_ID'] 21361
# ['q2_resonances', 'final_state_particle_IP'] 26196
# ['q2_resonances', 'Kstar_inv_mass'] 29704
# ['B0_vertex_chi2', 'DIRA'] 32630
# ['Kstar_inv_mass', 'DIRA'] 34870
# ['FD', 'DIRA'] 43664

# Output for combinations of 3 selectors
# ['q2_resonances', 'DIRA', 'Particle_ID'] 384
# ['q2_resonances', 'Kstar_inv_mass', 'DIRA'] 1389
# ['q2_resonances', 'B0_vertex_chi2', 'DIRA'] 1493
# ['q2_resonances', 'FD', 'Particle_ID'] 1863
# ['q2_resonances', 'final_state_particle_IP', 'DIRA'] 2080
# ['q2_resonances', 'FD', 'DIRA'] 2087
# ['q2_resonances', 'B0_vertex_chi2', 'Particle_ID'] 2156
# ['q2_resonances', 'B0_IP_chi2', 'DIRA'] 2273
# ['q2_resonances', 'final_state_particle_IP', 'Particle_ID'] 3425
# ['q2_resonances', 'Kstar_inv_mass', 'Particle_ID'] 3995