from core import B0_MM_selector, load_file, RAWFILES, combine_n_selectors
from ES_functions.Compiled import (q2_resonances, Kstar_inv_mass, B0_vertex_chi2,
 final_state_particle_IP, B0_IP_chi2, FD, DIRA, Particle_ID)
from histrogram_plots import generic_selector_plot, plot_hist_quantity
import matplotlib.pyplot as plt

both = combine_n_selectors(q2_resonances, B0_vertex_chi2, B0_MM_selector)

total_dataset = load_file(RAWFILES.TOTAL_DATASET)
s, ns = both(total_dataset, B0_vertex_chi2 = 0.2)

print(len(s),len(ns))

generic_selector_plot(total_dataset, s, ns, 'q2')
generic_selector_plot(total_dataset, s, ns, 'B0_ENDVERTEX_CHI2')
generic_selector_plot(total_dataset, s, ns, 'B0_MM')
