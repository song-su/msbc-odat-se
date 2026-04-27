MSBC-ODAT: Multi-Spectra Baseline Correction with Optimization
Overview

This project implements a Multi-Spectra Baseline Correction (MSBC) method based on the extension of Asymmetric Least Squares (ALS) for multiple spectra.

The original MSBC method was proposed in the paper:

Asymmetric least squares for multiple spectra baseline correction
Analytica Chimica Acta (2010)
DOI: https://doi.org/10.1016/j.aca.2010.08.033

The original implementation was developed in MATLAB.
In this project, the method is:

Reimplemented in Python

Enhanced with Bayesian optimization

Improved using Chauvenet’s criterion to mask peaks

Further integrated into the ODAT-SE optimization framework

Requirements
pip install numpy pandas matplotlib scipy bayesian-optimization odat-se[all]

Input data
spectra.xlsx

ODAT-SE Integration

The MSBC model is integrated into the ODAT-SE framework, enabling:

Global parameter search,
High-dimensional optimization

Advanced sampling strategies:
PAMC (Parallel Annealing Monte Carlo)
Bayesian search
Replica exchange

Standalone MSBC
python msbc_english.py

ODAT-SE Optimization
python msbc_odat_solver.py input_xxx.py

Output
Baseline-corrected spectra
Residuals:

(raw−baseline)/y0

Optimization logs

Visualization plots
