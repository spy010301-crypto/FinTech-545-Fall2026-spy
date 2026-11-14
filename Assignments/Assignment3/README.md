# Assignment 3 - Options, Option Risk, and Portfolio Construction

Start with `Assignment3_Responses.pdf`. The first five pages answer one problem
per page using **Predict, Fit, Reconcile**. The appendices contain every quote's
IV/vega, complete regression coefficients/SEs, and the requested plots. Numbers
are listed in short paragraphs rather than tables. `Predictions.md` preserves
the initial expectations, including predictions that need qualification later.

## Install and run

From the course repository:

```bash
cd Assignments/Assignment3
```

From the separate assignment repository:

```bash
cd Assignment3
```

Then use Python 3.10 or newer:

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

This runs all five problems, saves detailed unrounded results in `results/`,
creates the figures, and rebuilds both response files. No Julia, notebook,
network download, separate course checkout, or expected-output file is needed.
All data inputs are included in this folder. Paths are relative to the scripts,
so launching a script from another directory also works.

To rerun one problem or rebuild only the responses:

```bash
python3 problem1.py
python3 problem2.py
python3 problem3.py
python3 problem4.py
python3 problem5.py
python3 build_responses.py
```

Problem 3 uses 1,000,000 ten-day paths per distribution and seed 1234.
Problem 4 uses 500 six-month-block bootstrap samples and seed 545. Runs normally
take less than a few minutes; timing depends on the computer. Simulation numbers
are estimates and may differ slightly with numerical package versions.

## Where the code is

- `problem1.py`: parity, all bid/mid/ask IVs, vega and smile plot.
- `problem2.py`: four option prices, exact-dividend trees, convergence, Greeks,
  dividend sensitivities and call exercise boundary.
- `problem3.py`: return fits, Greeks, delta-normal risk, full revaluation,
  delta-gamma/theta comparisons and P&L curves.
- `problem4.py`: asset moments, three tangency portfolios, frontier/CML,
  tail risk, downside CAPM and weight stability.
- `problem5.py`: CAPM/FF3/Carhart regressions, factor-premium expectations,
  momentum stress and alpha reconciliation.
- `risk_library.py`: reusable GBSM, Greeks, IV inversion, distribution fitting,
  risk measures, book valuation and stock simulation.
- `american_options.py`: reusable CRR tree and central-difference Greeks.
- `portfolio_library.py`: reusable OLS, tangency weights, efficient frontier,
  and fractional empirical ES sensitivity.
- `helpers.py`: assignment paths and JSON writing only.
- `run_all.py`: executes the workflow. `build_responses.py` writes the PDF/Markdown.

## Reuse the Python library

The three calculation modules are a reusable Python library, not just scripts
that solve these particular CSVs. Their imports do not read data, fit models,
print results, or create files. Put the modules beside another script or add
this directory to your Python path. For example:

```python
from risk_library import gbsm, option_greeks, historical_risk
from american_options import binomial_price

price = gbsm('call', 100, 100, 0.5, 0.04, 0.25)
greeks = option_greeks('call', 100, 100, 0.5, 0.04, 0.25)
american_put = binomial_price('put', 100, 100, 0.5, 0.04, 0.25, 100)
risk = historical_risk([-10, -4, 2, 5, 9], alpha=0.20)
```

GBSM takes annual decimal rates/volatility, years to expiry and an optional
continuous dividend yield. The generalized stock carry is `rate-dividend_yield`.
Vega is per 1.00 volatility, and theta is per elapsed year. Tree option types
are lowercase `call`/`put`; dividend dates must lie exactly on the step grid.
The tree starts independent subtrees after cash dividends, with no interpolation
or forced recombination. For arbitrary inputs, its stock floor caps a dividend
at the available stock value; that floor does not bind in this assignment.

Portfolio means, covariance and risk-free rates must use the same frequency.
`maximum_sharpe` requires a positive normalization for a finite unconstrained
positive-Sharpe tangency portfolio; two of the 500 optional unrestricted
bootstrap samples fail that condition and are explicitly excluded. Constrained
bootstrap samples are all retained. OLS SEs are conventional homoskedastic SEs.

## Results and conventions

Every problem saves `results/problemN_results.json`. Additional CSVs contain
the full parity/IV outputs, P&L curves, frontier weights, bootstrap weights,
regression estimates and factor premia. The PDF rounds these results. The
response builder formats current results and contains short interpretation
text; if the supplied data change, recheck that interpretation and its example
comparisons rather than assuming the same conclusions still hold.

- Original question PDF and seven CSV data files are preserved unchanged.
- Problem 1 uses calendar days/365; Problem 3 uses trading days/255 and the
  original single-share option holdings (no extra contract multiplier).
- Normal/t fits include a fitted location. Problem 3 sets daily simulation
  means to zero while preserving the fitted scale/shape; option IV stays fixed.
  The arithmetic distributions have unbounded downside. This seeded run has no
  return at or below -100%; code raises if one occurs instead of clipping it.
- The delta-normal calculation uses sample daily SD, zero mean P&L and no
  gamma/theta; both nonlinear approximations are separately compared on the
  same full-revaluation scenarios. There is no dynamic rehedging or cash account.
- Moments use sample variance and central skew/excess kurtosis. Historical
  VaR/ES follow `library/RiskStats.jl`: average the floor/ceiling order statistics
  at `alpha*n` using 1-based indexing, then average the tail at/below the cutoff.
  At 1% of 240 months, this uses two observations for ES. A fractional-tail ES
  sensitivity is separately labelled. Positive risk numbers denote losses.
- Problem 4 uses monthly RF=0.03/12, mean times 12 and SD times sqrt(12).
  Its supplied exercise data end 2026-11-30; they are not represented as a live
  realized performance history. Ordinary mean SE assumes independent months;
  the weight bootstrap preserves six-month stretches and cross-asset rows.
- Problem 5 converts all factors and RF from percent to decimal, and joins
  dates one-to-one. The latest factor decade is (2014-09-30, 2024-09-30], not
  the 1996-2001 fund sample. This is a hypothetical premium scenario, not a
  backtest. Annual expected and realized comparisons use 255 times daily means;
  CAGR is saved separately. AICc includes the residual variance parameter.

## Course references and verification

Methods follow the supplied Week 5 risk measures, Week 6 GBSM/parity, Week 7
American options/Greeks/portfolio construction, and Week 9 factor-regression
examples, including `library/gbsm.jl`, `bt_american.jl`, `RiskStats.jl` and
`optimizers.jl`. They are translated to Python, not replaced with an option
pricing package. External solutions or expected-output files are not read.

Independent checks covered IV repricing and missing-IV bounds, analytic Greeks
against finite differences, dividend trees against scalar/full-path recursions,
all four books' simulated P&L and risk, OLS against QR/statsmodels, and tangency
weights from multiple starts. The optional unrestricted bootstrap edge case was
fixed before delivery. The full workflow was also rerun from an input-only copy;
all numerical JSON/CSV outputs reproduced exactly. PDF pages and plots were
visually checked. Review scripts remain outside the submission to keep it simple.

Verified runtime: Python 3.13; NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.2,
Matplotlib 3.10.6 and ReportLab 4.4.9.

The question also asks you to enter `done` in Canvas after submitting the
repository. Running this code or pushing GitHub does not perform that final step.
