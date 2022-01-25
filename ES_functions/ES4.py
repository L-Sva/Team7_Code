from core import load_file
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18

def filter_B0_chi(dataframe):
    accept = dataframe[dataframe['B0_IPCHI2_OWNPV'] < 9]
    reject = dataframe[dataframe['B0_IPCHI2_OWNPV'] >= 9]
    return accept, reject

if __name__ == '__main__':
    total_dataset = load_file('total_dataset.pkl')
    accept, reject = filter_B0_chi(total_dataset)
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(total_dataset['B0_IPCHI2_OWNPV'], bins=100, label='total_dataset')
    ax.axvline(9, c='r', ls='--', label='Upper cut-off')
    ax.set(xlabel='$B_0$ $\chi^2$ value', ylabel='Number of events')
    ax.legend(prop={'size': 14})
    plt.show()

