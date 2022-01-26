import matplotlib.pyplot as plt
from histrogram_plots import plot_hist_quantity

def plot_acceptance_quantities(df,axs = None, label=None):
    if axs is None:
        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
    plt.sca(axs[0])
    plot_hist_quantity(df,'q2',label=label)
    plt.sca(axs[1])
    plot_hist_quantity(df,'costhetal',label=label)
    plt.sca(axs[2])
    plot_hist_quantity(df, 'costhetak',label=label)
    plt.sca(axs[3])
    plot_hist_quantity(df, 'phi',label=label)
    return axs

if __name__ == '__main__':

    from core import RAWFILES, load_file
    from ES_functions.ES1 import ES1

    acceptance = load_file(RAWFILES.ACCEPTANCE)

    axs = plot_acceptance_quantities(acceptance, label='No select')
    subset, not_subset = ES1(acceptance)
    plot_acceptance_quantities(subset, axs=axs, label='ES1')
    plt.legend()
    plt.tight_layout()
    plt.show()