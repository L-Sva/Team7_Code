# TBPS github

In order to do data analysis, you must download all .pkl files from https://imperialcollegelondon.app.box.com/s/mwdgg4uz7hdz56bx6w4loc04qvzb7tmy and place them in the `/data/` folder.
These files are too large to upload to github.

By default .pkl will not be uploaded to github when you upload your code.

Code to load and save .pkl files, as well as an example selector function, are in `core.py` .
Example of how to use this is in `example_of_using_core.py` .

## Running on college PC in console (using Anaconda on apps anywhere):

- Copy the project to somewhere on H: drive
- Copy the directory `bayes_opt` from https://github.com/fmfn/BayesianOptimization into the project root directory
- Open Anaconda Navigator
- Open console_shortcut
- Navigate to the directory with your python files (using `h:` and `cd /path/to/dir`)
- IMPORTANT: Updata pandas - `conda update pandas` - this fixes bugs in the python install IT has decided to use...
- Install xgboost - `conda install xgboost`
- Run via `python filename.py`
