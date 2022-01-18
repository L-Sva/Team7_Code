import pandas as pd
import os
import os.path as path
import pickle


def load_file(filename='total_dataset.pkl', folder='data') -> pd.DataFrame:
    res = None
    with open(path.join(folder,filename),'rb') as file:
        res = pickle.load(file)
    return res

def save_file(dataframe ,filename, foldername):
    if not path.exists(foldername):
        os.mkdir(foldername)
    with open(path.join(foldername, filename),'wb') as file:
        pickle.dump(dataframe,file)
    
def example_selector(dataframe):
    subset = dataframe[dataframe['B0_MM'] < 5350]
    not_subset = dataframe[dataframe['B0_MM'] > 5350]
    return subset, not_subset

