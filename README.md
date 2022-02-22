# TBPS github

In order to do data analysis, you must download all .pkl files from https://imperialcollegelondon.app.box.com/s/mwdgg4uz7hdz56bx6w4loc04qvzb7tmy and place them in the `/data/` folder.
These files are too large to upload to github.

By default .pkl will not be uploaded to github when you upload your code.

Code to load and save .pkl files, as well as an example selector function, are in `core.py` .
Example of how to use this is in `example_of_using_core.py` .

In order to import and run the machine learning selector, use the code
```python
    from ml_selector import remove_all_bks
    from core import load_file, RAWFILES
    data = load_file(RAWFILES.TOTAL_DATASET)
    subset, notsubset = remove_all_bk(data)
```

## File descriptions
```
    acceptance_func_parameters

    data
        Raw data files as give by Mitesh
    data_combinatorial_background_sample_histograms
        Histograms showing the properties of the data selected for ML training 
        on the combinatorial background
    data_histograms
        Histograms of total_dataset vs. signal
    data_histograms_with_jpsi
        Histograms of total_dataset vs. signal vs. jpsi
    ES_functions
        Manual selector functions
    ml_models
        Saved machine learning models
    optimisation_models_comb
        Saved machine learning models from hyperparameter optimisation for 
        combinatorial background
    optimisation_models_peaking
        Saved machine learning models from hyperparameter optimisation for 
        peaking background
    _ml_histograms_on_total
        Histograms of main ML selector run on total_dataset
    _ml_hist_individual_bks
        Histograms of main ML selector run on individual backgrounds
    acceptance_plot_tool.py
        Plots histograms in the angular quantities
    binning.py
        Binning for ???
    core.py
        Code to load raw data
    example_combining_arbitrary_selectors.py
        Code to combine N selectors of choice
    example_of_using_core.py
        duh
    histrogram_plots.py
        Code to make histogram plots
    ml_bivariant_example.py
        Simple illustrative example of benefit of ML as a multivariate method
    ml_combinatorial_extraction.py
        Code for loading combinatorial background training data
    ml_count_bk_in_total.py
        Code for estimating remaining peaking background events after main ML 
        selector
    ml_recreate.py

    ml_selector.py
    ml_tools.py
    ml_train.py
    selection_cuts_hist.py
    starter_notebook-Copy1.ipynb
    starter_notebook.ipynb
    Summed_dataset2d_ratio_Legendre_polinomial.py
    test_candidates_example.py
```
## Running on college PC in console (using Anaconda on apps anywhere):

- Copy the project to somewhere on H: drive
- (If using bayesian optimisation) Copy the directory `bayes_opt` from https://github.com/fmfn/BayesianOptimization into the project root directory
- Open Anaconda Navigator
- Open console_shortcut
- Navigate to the directory with your python files (using `h:` and `cd /path/to/dir`)
- IMPORTANT: Updata pandas - `conda update pandas` - this fixes bugs in the python install IT has decided to use...
- Install xgboost - `conda install xgboost`
- Run via `python filename.py`
