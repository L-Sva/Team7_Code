import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os.path as path
import mplhep

# Requires all .pkl files provided to be placed in /data/

def load_file(filename='total_dataset.pkl') -> pd.DataFrame:
    res = None
    with open(path.join('data',filename),'rb') as file:
        res = pickle.load(file)
    return res


def plot_hist_quantity(df,column,bins=100,range=None,label=None):
    # Prevents exception being thrown for 'year' column
    if df[column].dtype != 'object':
        h, bins = np.histogram(df[column], bins=bins, range=range)
        h_density, _ = np.histogram(df[column], bins=bins, density = True)
        half_binwidths = (bins[1] - bins[0]) / 2
        bin_centres = (bins[1:] + bins[:-1]) / 2
        #mplhep.histplot(h, bins, yerr=True, histtype='errorbar', xerr=half_binwidths, marker='.')
        plt.errorbar(
            x=bin_centres, y=h_density, 
            yerr=np.sqrt(h) / h * h_density, 
            xerr=half_binwidths, capsize=3, ls='', label=label)
        plt.xlabel(column)
        plt.ylabel('Probability density')
        plt.xlim(bins[0], bins[-1])

        return bins, h

if __name__ == '__main__':
    total_dataset = load_file('total_dataset.pkl')
    jpsi = load_file('jpsi.pkl')
    signal = load_file('signal.pkl')

    for column in total_dataset:
        bins, h = plot_hist_quantity(total_dataset, column, label='total_dataset')
        plot_hist_quantity(jpsi, column, label='jpsi', bins=bins)
        #plot_hist_quantity(signal, column, label='signal', bins = bins)
        plt.legend()
        #plt.show()
        plt.savefig(path.join('data_histograms',f'{column}.png'))
        plt.close()