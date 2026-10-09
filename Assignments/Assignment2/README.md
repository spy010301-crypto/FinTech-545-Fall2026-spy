# Assignment 2

This self-contained folder contains the original questions/data, simple Python
programs for all five problems, figures, unrounded results, and written responses.
`Assignment2_Responses.pdf` has one problem per page. Each answer follows
Predict, Fit, Reconcile. The pre-fit predictions are also kept in `Predictions.md`.

## Install and run

First enter this folder from the repository root. In the course repository:

```bash
cd Assignments/Assignment2
```

In the separate assignment repository:

```bash
cd Assignment2
```

Then install the packages and run:

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

`run_all.py` first makes the raw-data plots, then runs Problems 1-5. Every problem
prints its results and saves unrounded numbers in `results/`. Simulations use
100,000 draws with seed 1234. Small numerical differences across package versions
are possible; Monte Carlo results are estimates, not exact population quantiles.

To run one problem, for example:

```bash
python3 problem4.py
```

To rebuild the PDF from the readable written-answer source:

```bash
python3 build_responses.py
```

The numerical results are computed from the CSVs. The written explanation is
stored separately in `Assignment2_Responses.md`: if the inputs change, update its
numbers/explanation before rebuilding the PDF. No other repository or Julia
installation is needed. All paths are relative to the Python files, so the
programs also work when launched from a different directory.

## Programs and outputs

- `problem1.py`: jointly observed counts, both correlations/eigenvalues,
  Cholesky attempts, tracking variance, Rebonato-Jackel and Higham repairs.
- `problem2.py`: arithmetic/demeaned returns, EW weights, effective sample
  sizes/half-lives, regime SDs, normal/t/historical VaRs, rough estimation errors.
- `problem3.py`: bond moments, default counts, historical VaR/ES at 5% and 1%,
  normal VaR for all four positions.
- `problem4.py`: normal/t margin selection, Kendall correlation, Gaussian/t
  copula fitting, simulation risk, joint-tail counts, daily likelihood gains,
  and tail dependence.
- `problem5.py`: OLS market models, residual covariance/correlation, both
  simulations, and direct delta-normal VaR.
- `helpers.py`: a few shared calculations, adapted from the course examples.
- `preview_inputs.py`: saves the required plots in `figures/` without model fitting.
- `results/problem1_results.json` through `problem5_results.json`: every
  unrounded calculation reported in the responses, including full matrices.
- `results/problem4_daily_likelihood.csv`: every day's Gaussian/t log density,
  difference, fitted uniforms, and empirical-rank middle-day indicator.

## Statistical conventions

- Arithmetic returns are `current_price / previous_price - 1`.
- Moments use sample variance (`n-1`) and standardized central skewness/excess
  kurtosis, as in Week 1. Normal population excess kurtosis is zero.
- Historical risk follows `library/RiskStats.jl`: average the floor/ceiling
  order statistics at `alpha*n`, with 1-based indexing; ES is minus the average
  P&L at or below that cutoff. Here all required tail sizes are integers and no
  cutoff ties occur. Negative VaR represents a gain and is not floored or made absolute.
- Problem 1 uses pair-specific Pearson correlations but each asset's full-history
  sample SD when constructing the requested tracking covariance. Repairs act on
  the correlation matrix with a fixed unit diagonal.
- Problem 2 uses normalized finite EW weights, newest observation first in
  importance, and weighted centering as in `library/ewCov.jl`. All normal VaRs
  are zero-mean; the fitted t retains its estimated location after sample demeaning.
- Problem 3 uses the given scenario returns directly: no further 100/90
  multiplication, and no demeaning of the bond P&L.
- Problem 4 normal and t margins use MLE; normal MLE scale uses `n`, unlike the
  descriptive sample SD. AICc counts 2 normal-margin and 3 t-margin parameters.
  Both copulas use the same frozen Kendall `R = sin(pi*tau/2)`. Following Week 5,
  copula AICc/BIC count 0 new Gaussian parameters and 1 new t parameter (`nu`).
  The t copula profiles `2.0001 <= nu <= 100`, as in the course, independently of
  the t-margin degrees of freedom. Joint simulated tails use population uniform
  cutoffs 0.025/0.975. Part (i)'s central days use observed ranks, consistent with
  the pre-fit tail counts; the JSON also gives the fitted-CDF alternative.
- Problem 5 includes an OLS intercept when fitting but omits alphas in simulation
  to honor zero expected returns. Market, stock, and residual covariance all use
  `n-1`, which preserves the OLS covariance identity. The two residual simulations
  reuse the same normal draws.

The methods follow the supplied Week 1-5 notes and `library/missing_cov.jl`,
`ewCov.jl`, `fitted_model.jl`, `RiskStats.jl`, `copula.jl`, and
`multivariate_t.jl`, translated into Python. Only `problem1.csv`-`problem5.csv`
are inputs; no expected-output files or precomputed answer numbers are read.

Verification: an isolated rerun containing only the five CSV inputs and calculation
programs reproduced all five JSON files and the daily likelihood CSV exactly.
Independent checks covered tail order statistics, EW effective sample sizes,
copula log densities, and the OLS covariance identity. The PDF was checked page
by page, and 103 displayed matrix/eigenvalue/risk numbers were compared with
the results. Numerical runtime: NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.2;
figures: Matplotlib 3.10.6; PDF: ReportLab 4.4.9.

The assignment also asks for a repository upload and a Canvas response of
`done`. These are separate submission steps, not actions performed by the code.
