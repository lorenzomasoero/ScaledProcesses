This repository contains code and data to replicate the experiments and plots in "Scaled process priors for Bayesian nonparametric estimation of the unseen genetic variation", (https://www.tandfonline.com/doi/full/10.1080/01621459.2022.2115918, https://arxiv.org/abs/2106.15480). 

# Structure

The repository is divided into 4 main folders:
* `utils_folder/` which contains all the code and functions to replicate the analysis. In particular, each method considered has its own `.py` file.
* `Synthetic/` which contains example usage on synthetic datasets
* `Cancer/` which contains data and code to run and fit models on TCGA cancer data, and reproduce plots.
* `gnomAD/` which contains data and code to run and fit models on the gnomAD dataset, and reproduce plots.

```
|____gnomAD
| |____Plots.ipynb
| |____Fit.ipynb
| |____results
| |____data
| |____Plots
|____Cancer
| |____Plots.ipynb
| |____Fit.ipynb
| |____results
| |____minimal_results_plots.ipynb
| |____Data
| |____Plots
|____Synthetic
| |____Fit.ipynb
| |____results
| |____Plots
|____utils_folder
```

# Data description

Both `Cancer/` and `gnomAD/` contain data to fit the models and reproduce the analysis. In particular:

* `Cancer/Data/TCGA` contains 33 datasets from the [TCGA project](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). 
Each dataset refers to a specific [cancer type](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations), and is a binary matrix with shape `N, K` where `N` is the number of patients in the dataset with cancer of the specific type, and `K` is a gene, targeted by cancer. The `(n,k)` entry is equal to 1 if patient `n` showed variation within gene `k`. Additional details on this data are discussed in Appendix F of "[More for less: Predicting and maximizing genetic variant discovery via Bayesian nonparametrics" (Masoero et al., Biometrika 2022)](https://arxiv.org/pdf/1912.05516.pdf).
* `gnomAD/data` contains 15 folders, each folder referring to a different subpopulation in the data collected by the [gnomAD project](https://gnomad.broadinstitute.org/news/2018-10-gnomad-v2-1/). Each folder contains data about the subpopulation organized in three subfolders 
	* `cts/` contains four datasets: 
		* `all.txt`, a single accumulation curve for the subpopulation, which at position `n` (for a fixed ordering of all the individuals samples) the total number of distinct variants observed in the first `n` individuals
		* `N_50.npy`, an array of size `(50,51)`. Each row of this array is an accumulation curve for the population under study, obtained by retaining a random subset of 50 samples (without replacement) from the subpopulation. The first value (first column) is 0 by construction.
		* `N_100.npy`, an array of size `(50,101)`. As above but now for 100 random samples.
		* `N_200.npy`, an array of size `(50,201)`. As above but now for 200 random samples.
	* `sfs/` contains three datasets:
		* `N_50.npy`, an array of size `(50,51)`. Each row of this array is the site-frequency-spectrum for the population under study obtained by retaining a random subset of 50 samples from the population. The first entry (first column) is the number of variants not observed yet. Notice, the corresponding accumulation curve is the corresponding row in the `cts/` folder for the same file.
		* `N_100.npy`, as above but now for 100 random samples.
		* `N_200.npy`, as above but now for 200 random samples.

# Fitting

In `Cancer/`, `gnomAD/` and `Synthetic/` you will find `Fit.ipynb`, an iPythonNotebook which contains all the code needed in order to fit the experiments and save the data necessary to then reproduce the plots. Notice: `Synthetic/Fit.ipynb` also contains code to produce figures for the syntetic data. The relevant functions called to fit the methods can be found in the `utils/` folder.

# Plotting

In `Cancer/` and `gnomAD/` you will find `Plots.ipynb`, an iPythonNotebook which contains all the code needed in order to produce the plots displayed in the paper.
* `Cancer/Plots.ipynb` reproduces in the main text (Figures 1 -- 5).
* `Synthetic/Fit.ipynb` reproduces in Appendices F, G (Figures 6 -- 20).
* `gnomAD/Plots.ipynb` reproduces in Appendix H (Figures 21 -- 38).



