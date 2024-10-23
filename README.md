# ```Cosmo-ML-Tools```
<img src="docs/assets/logo.jpg" width="400" height="400">

<!-- ![image](docs/assets/logo.png) -->
### A Cosmology & Machine Learning Toolbox
[![image](https://img.shields.io/pypi/v/cosmo_ml_tools.svg)](https://pypi.python.org/pypi/cosmo-ml-tools)
[![image](https://img.shields.io/conda/vn/conda-forge/cosmo-ml-tools.svg)](https://anaconda.org/conda-forge/cosmo-ml-tools)


**An attempt to wrap many useful python packagages, ML algorithms and automate common workflows in Cosmology.**

**DISCLAIMER**: This is very much work in progress. It is originally intended for my own use, so the documentation might not be as good as I would like it to be.

-   Free software: MIT License
-   Documentation: https://rcalderonb6.github.io/cosmo-ml-tools
    
**Bare in mind that this code was written by a physicist! I am sure things can be made much more efficient/elegant.**

## Requirements
* `Matplotlib`
* `numpy`
* `jax`
* `pandas`
* `Getdist`

Optional requirements:
* `Class/hi_class` - Boltzmann solver for the computation of cosmological observables. 
* `Harmonic` - For Bayesian Evidence Computation and Model Selection
* `ChainConsumer` - For post-processing of the chains (Useful in some specific cases - Defaults to `Getdist`)
* `Numpyro` - For probabilistic programming, with support for auto-differentiation and Hamiltonian MonteCarlo (HMC) sampling schemes.
* `GPry` - For likelihood emulation using GP. Useful when likelihood computations are time-consuming!

## Features

-   TODO
