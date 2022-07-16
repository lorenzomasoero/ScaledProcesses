This repository contains code and data to replicate the experiments and plots in "Scaled process priors for Bayesian nonparametric estimation of the unseen genetic variation" (https://arxiv.org/abs/2106.15480). 

# Structure

The repository is divided into 4 main folders:
* `utils_folder` which contains all the code and functions to replicate the analysis. In particular, each method considered has its own `.py` file.
* `Final_Synthetic_Experiments` which contains example usage on synthetic datasets
* `Final_Cancer_Experiments` which contains data and code to run and fit models on TCGA cancer data, and reproduce plots.
* `Final_gnomAD` which contains data and code to run and fit models on the gnomAD dataset, and reproduce plots.

```
|____Final_gnomAD
| |____Plots.ipynb
| |____Fit.ipynb
| |____results
| |____data
| |____Plots
|____Final_Cancer_Experiments
| |____Plots.ipynb
| |____Fit.ipynb
| |____results
| |____minimal_results_plots.ipynb
| |____Data
| |____Plots
|____Final_Synthetic_Experiments
| |____Final_Fit_Plot.ipynb
| |____results
| |____Plots
|____utils_folder
```

# Data description

Both folders related to the real data applications contain data to fit the models and reproduce the analysis. In particular:

* `Final_Cancer_Experiments/data/TCGA` contains 33 datasets from the TCGA project (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). 
Each dataset refers to a specific [cancer type>>https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations], and is a binary matrix with shape `N, K` where `N` is the number of patients in the dataset with cancer of the specific type, and `K` is a gene, targeted by cancer. The `(n,k)` entry is equal to 1 if patient `n` showed variation within gene `k`. Additional details on this data are discussed in Appendix F of "More for less: Predicting and maximizing genetic variant discovery via Bayesian nonparametrics" (Masoero et al., https://arxiv.org/pdf/1912.05516.pdf).
* `Final_gnomAD/data` contains 15 folders, each folder referring to a different subpopulation in the data collected by the gnomAD project (see https://gnomad.broadinstitute.org/news/2018-10-gnomad-v2-1/ for population informations). Each folder contains data about the subpopulation organized in three subfolders 
	* `cts/` contains four datasets: 
		* `all.txt`, a single accumulation curve for the subpopulation, which at position `n` (for a fixed ordering of all the individuals samples) the total number of distinct variants observed in the first `n` individuals
		* `N_50.npy`, an array of size `(50,51)`. Each row of this array is an accumulation curve for the population under study, obtained by retaining a random subset of 50 samples from the population. The first value (first column) is 0 by construction.
		* `N_100.npy`, an array of size `(50,101)`. Each row of this array is an accumulation curve for the population under study, obtained by retaining a random subset of 100 samples from the population. The first value (first column) is 0 by construction.
		* `N_200.npy`, an array of size `(50,201)`. Each row of this array is an accumulation curve for the population under study, obtained by retaining a random subset of 200 samples from the population. The first value (first column) is 0 by construction.
** `sfs/` contains three datasets:
*** `N_50.npy`, an array of size `(50,51)`. Each row of this array is the site-frequency-spectrum for the population under study obtained by retaining a random subset of 50 samples from the population. The first entry (first column) is the number of variants not observed yet. Notice, the corresponding accumulation curve is the corresponding row in the `cts/` folder for the same file.
*** `N_100.npy`, an array of size `(50,51)`. Each row of this array is the site-frequency-spectrum for the population under study obtained by retaining a random subset of 100 samples from the population. The first entry (first column) is the number of variants not observed yet. Notice, the corresponding accumulation curve is the corresponding row in the `cts/` folder for the same file.
*** `N_200.npy`, an array of size `(50,51)`. Each row of this array is the site-frequency-spectrum for the population under study obtained by retaining a random subset of 200 samples from the population. The first entry (first column) is the number of variants not observed yet. Notice, the corresponding accumulation curve is the corresponding row in the `cts/` folder for the same file.

# Fitting

You will find Jupyter notebooks in the `examples/` folder that reproduce the visualizations in the documentation.
In Jupyter lab/notebook, run `Kernel -> Restart & Run All` to run the notebooks. Note: currently the notebooks
are coded such that they must be run in linear, top-to-bottom order (hence `Kernel -> Restart & Run All`). 

# Plotting







