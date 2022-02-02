from core import load_file, RAWFILES
from histrogram_plots import generic_selector_plot, plot_hist_quantity
import matplotlib.pyplot as plt
from os import path
import ml_tools

def B0_MM_selector(dataset, B0_MM = 5400):
    accept = dataset['B0_MM'] > B0_MM
    s = dataset[accept]
    ns = dataset[~accept]
    return s, ns

def load_combinatorial_train_validate_test(train_samples_limit = None, remove_columns = True):
    signal_data = load_file(RAWFILES.SIGNAL)
    bk_data, _ = B0_MM_selector(load_file(RAWFILES.TOTAL_DATASET), 5400)

    sig_train, sig_validate, sig_test = ml_tools.ml_prepare_train_validate_test(signal_data)
    bks_train, bks_validate, bks_test = ml_tools.ml_prepare_train_validate_test(bk_data)

    reject_column_names = ()
    if remove_columns:
            reject_column_names = ('B0_MM','Kstar_MM')

    for df in [sig_train, sig_validate, sig_test, bks_train, bks_validate, bks_test]:
        ml_tools.ml_strip_columns(df, inplace=True, reject_column_names=reject_column_names)

    train_data = ml_tools.ml_combine_signal_bk(sig_train[:train_samples_limit], bks_train[:train_samples_limit])
    validate_data = ml_tools.ml_combine_signal_bk(sig_validate, bks_validate)
    test_data = ml_tools.ml_combine_signal_bk(sig_test, bks_test)

    return train_data, validate_data, test_data


if __name__ == '__main__':
    IMAGE_OUTPUT_DIR = 'data_combinatorial_background_sample_histograms'
    SAVEFIG = False

    total_dataset = load_file(RAWFILES.TOTAL_DATASET)
    signal = load_file(RAWFILES.SIGNAL)

    total_s, total_ns = B0_MM_selector(total_dataset)
    signal_s, signal_ns = B0_MM_selector(signal)

    n1 = 100 /len(total_dataset)
    n2 = 100 /len(signal)
    print(f'Total dataset | {len(total_s)} accepted, {len(total_ns)} rejected')
    print(f'Signal dataset | {len(signal_s)} accepted, {len(signal_ns)} rejected')
    print(f'Total dataset | {len(total_s)*n1:.3g}% accepted, {len(total_ns)*n1:.3g}% rejected')
    print(f'Signal dataset | {len(signal_s)*n2 :.3g}% accepted, {len(signal_ns)*n2:.3g}% rejected')

    if SAVEFIG:
        for column in total_dataset:
            if total_dataset[column].dtype != 'object':
                bins, h = plot_hist_quantity(total_dataset, column, label='Total', bins=150)
                plot_hist_quantity(total_s, column, label='BK subset', bins = bins)
                plot_hist_quantity(total_ns, column, label='Not BK subset', bins=bins)
                plot_hist_quantity(signal, column, label='Signal', bins=bins)
                plt.legend()

                plt.savefig(
                    path.join(IMAGE_OUTPUT_DIR,f'{column}.png')
                )
                plt.close()

    generic_selector_plot(total_dataset, total_s, total_ns, 'B0_MM')
    generic_selector_plot(signal, signal_s, signal_ns, 'B0_MM')