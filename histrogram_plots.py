import warnings
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from core import load_file

# Requires all .pkl files provided to be placed in /data/

def plot_hist_quantity(df,column,bins=100,range=None,label=None,color='black'):
    # Prevents exception being thrown for 'year' column
    if df[column].dtype != 'object':

        # Generating histogram data
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            h, bins = np.histogram(df[column], bins=bins, range=range)
            h_density, _ = np.histogram(df[column], bins=bins, density = True)
            half_binwidths = (bins[1] - bins[0]) / 2
            bin_centres = (bins[1:] + bins[:-1]) / 2
            yerr = np.divide(np.sqrt(h), h, out = np.zeros(h.shape, dtype='float'), where = h!=0) * h_density
            plt.errorbar(
                x=bin_centres, y=h_density,
                # Need to use this to get correct sized errorbars on a histogram with density=True
                # yerr = 0 for h = 0
                yerr= yerr,
                xerr=half_binwidths, capsize=3, ls='', label=label, color=color)
        plt.xlabel(column)
        plt.ylabel('Probability density')
        plt.xlim(bins[0], bins[-1])

        return bins, h, h_density
    else:
        warnings.warn(f'Cannot plot column {column}, as it has dtype of `object`')

def generic_selector_plot(orginal,subset, not_subset, column, bins = 100, show = True):
    bins, h = plot_hist_quantity(orginal, column, label='Original', bins=bins)
    plot_hist_quantity(subset, column, label='Subset', bins = bins)
    plot_hist_quantity(not_subset, column, label='Not subset', bins=bins)
    plt.legend()
    if show:
        plt.show()

def validPlotColumns(dataset):
    return [column for column in dataset if dataset[column].dtype != 'object']

if __name__ == '__main__':
    total_dataset = load_file('total_dataset.pkl')
    jpsi = load_file('jpsi.pkl')
    signal = load_file('signal.pkl')

    for column in total_dataset:
        if total_dataset[column].dtype != 'object':
            bins, h = plot_hist_quantity(total_dataset, column, label='total_dataset')
            plot_hist_quantity(signal, column, label='signal', bins=bins)
            plt.legend()
            plt.savefig(path.join('data_histograms',f'{column}.png'))
            plt.close()

    for column in total_dataset:
        if total_dataset[column].dtype != 'object':
            bins, h = plot_hist_quantity(total_dataset, column, label='total_dataset')
            plot_hist_quantity(signal, column, label='signal', bins=bins)
            plot_hist_quantity(jpsi, column, label='jpsi', bins=bins)
            plt.legend()
            plt.savefig(path.join('data_histograms_with_jpsi',f'{column}.png'))
            plt.close()