# Assignment 2

This self-contained folder contains the original questions/data, simple Python
programs for all five problems, figures, unrounded results, and written responses.
The reusable calculation library is `risk_library.py`; all five problem scripts
import it. `helpers.py` only handles this assignment's paths and JSON output.
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
- `risk_library.py`: reusable calculation functions, independent of assignment files.
- `helpers.py`: this assignment's folder path and JSON output only.
- `preview_inputs.py`: saves the required plots in `figures/` without model fitting.
- `results/problem1_results.json` through `problem5_results.json`: every
  unrounded calculation reported in the responses, including full matrices.
- `results/problem4_daily_likelihood.csv`: every day's Gaussian/t log density,
  difference, fitted uniforms, and empirical-rank middle-day indicator.

## Reuse the Python library

`risk_library.py` is a standalone Python module, not a program that runs a
particular problem. Its functions accept data and return results; importing it
does not read CSVs, print answers, create figures, or write files. No package
publishing or special installation is needed beyond the dependencies above.

For example, run this from this folder, or put `risk_library.py` beside your own
Python script:

```python
import numpy as np
from risk_library import moments, historical_risk

returns = np.array([-0.03, -0.02, 0.01, 0.04])
print(moments(returns))
print(historical_risk(1_000_000 * returns, alpha=0.25))
```

The main functions are:

- `arithmetic_returns(prices)` and `moments(values)` for returns and statistics.
- `exponential_weights(observations, decay)` and `ew_volatility(values, weights)`
  for normalized EW weights and weighted volatility.
- `correlation_to_covariance(correlation, standard_deviations)`,
  `portfolio_variance(covariance, weights)`, `near_psd(correlation)`, and
  `higham(correlation)` for covariance, portfolio variance, and correlation repairs.
- `historical_risk(pnl, alpha)`, `normal_var(pnl, alpha)`, and
  `delta_normal_var(covariance, weights, alpha)` for historical or normal risk.
- `aicc(log_likelihood, parameters, observations)` and `fit_margin(values)`
  for model comparison. `fit_margin` returns the chosen SciPy distribution and
  a dictionary of both fits' parameters and AICc values.
- `copula_correlation(uniforms)`, `copula_log_density(uniforms, correlation, nu)`,
  `fit_t_copula(uniforms, correlation)`, and `simulate_copulas(correlation, nu)`
  for copula fitting and simulation. `nu=None` selects the Gaussian log density.
- `joint_tail_counts(uniforms, asset_names, alpha, days)` and
  `copula_tail_dependence(rho, nu)` for joint tails.
- `fit_market_model(market, stocks)` returns alphas, betas, and residuals;
  `simulate_market_model(betas, market_variance, residual_covariance, draws)`
  returns zero-mean simulated stock returns.

Matrix inputs are NumPy arrays with days in rows and assets in columns. Keep
asset names, weights, standard deviations, and covariance columns in the same
order. Copula functions support different numbers of assets, not just this
assignment's three. `arithmetic_returns` takes a pandas Series or DataFrame;
`normal_var` takes a pandas Series or NumPy array. EW weights are oldest to newest
and sum to one. PSD repairs operate on correlation matrices, not covariances.
Copula fitting expects uniforms strictly inside (0, 1) and a positive-definite R;
`simulate_market_model` takes independent standard-normal draws with the market
in column 0 and the stock errors in the remaining columns. Normal risk functions
preserve the different mean conventions documented below.

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

After extracting the reusable library, input-only reruns before and after the
change reproduced all five JSON files, the daily likelihood CSV, and all three
figures exactly. Additional library-only checks used new small inputs and 2/4
assets, without any assignment CSV or expected-output files. The written answers
and PDF were not changed by this code reorganization.

The assignment also asks for a repository upload and a Canvas response of
`done`. These are separate submission steps, not actions performed by the code.
