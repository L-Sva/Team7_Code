# Checking the example selector in core works

from histrogram_plots import plot_hist_quantity
from core import load_file, B0_MM_selector, save_file
import matplotlib.pyplot as plt

# Allows file to be loaded as a module to allow reuse of functions defined above this line
# Do not define gloabal vars above this line
if __name__ == '__main__':

    # Load the datasets
    total_dataset = load_file('total_dataset.pkl')

    # Example of selecting events with 'B0_MM' above or below a cutoff
    below_mass_cutoff, above_mass_cutoff = B0_MM_selector(total_dataset)

    bins, h = plot_hist_quantity(total_dataset, 'B0_MM', label='total_dataset')

    # Pass the bins parameter to use the same bins for all datasets plotted on a histograms
    plot_hist_quantity(below_mass_cutoff, 'B0_MM',label='total_dataset_below_mass',bins=bins)

    plot_hist_quantity(above_mass_cutoff, 'B0_MM',label='total_dataset_above_mass',bins=bins)

    plt.legend()
    plt.show()

    save_file(below_mass_cutoff, 'total_dataset_below_mass.pkl','examples_save')

    test_load = load_file('total_dataset_below_mass.pkl','examples_save')

    plot_hist_quantity(test_load, 'B0_MM',bins=bins)
    plt.show()