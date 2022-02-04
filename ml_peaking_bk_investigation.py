
from ml_selector import remove_peaking_background
from core import load_file, RAWFILES, ensure_dir
import ml_tools
from histrogram_plots import generic_selector_plot, validPlotColumns
import os
import matplotlib.pyplot as plt

DIR ='ml_hist_individual_bks'
OUTPUT_PLOTS = False

individual_peak_bks = RAWFILES.peaking_bks

for file in RAWFILES.peaking_bks:
    IMG_DIR = os.path.join(DIR, file[:-4])

    ensure_dir(IMG_DIR)

    train, test = ml_tools.ml_prepare_train_test( load_file(file) )
    s, ns = remove_peaking_background(test)
    num = len(test)
    print(f'{file} | accepted: {len(s)} ({len(s)/num}), rejected {len(ns)} ({len(ns)/num})')
    if OUTPUT_PLOTS:
        cols = validPlotColumns(test)
        N = len(cols)
        i = 1
        for col in cols:
            generic_selector_plot(test, s, ns, col, 150, False)
            plt.title(f'ML peaking bk removal on {file[:-4]}')
            plt.savefig(os.path.join(IMG_DIR, f'{file[:-4]}_{col}.png'))
            plt.close()
            print(f'Outputing plots for {file} | {i} / {N}', end='\r')
            i += 1
        print('    Done: ')