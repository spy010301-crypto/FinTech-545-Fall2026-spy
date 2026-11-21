# Functional Tests 8.1–13.9 (Python)

This folder contains the second set of 25 course tests. The first set
(1.1–7.6) has its own program in [../python/](../python/).

## Run

From the course repository root:

```bash
python3 -m pip install -r testfiles/Python2/requirements.txt
python3 testfiles/Python2/run_tests.py
```

Verified with Python 3.13, NumPy 2.3.3, pandas 2.3.3, and SciPy 1.16.2.
The two cash-dividend trees in Test 12.3 do most of the work.

Each computed CSV is in [outputs/](outputs/).
[comparison.csv](outputs/comparison.csv) lists each test's status, method,
and maximum difference. A failed comparison produces a nonzero exit code.
Use `--output-dir` with a scratch directory to keep a separate run's results.

## Problems and program files

The test list is in [Tests.xlsx](../Tests.xlsx), Sheet1 rows 27–51.
The [data directory](../data/) contains both inputs and expected outputs.
[run_tests.py](run_tests.py) selects the inputs, calls the calculation
functions, and compares the outputs after calculating all 25 results.

- **8.1–8.6:** Normal and Student-t VaR/ES and simulated VaR/ES.
  Inputs: `test7_1.csv` for 8.1 and 8.4, `test7_2.csv` for the other four.
- **9.1:** Gaussian copula and 95% portfolio VaR/ES at asset and total levels.
  Inputs: `test9_1_returns.csv` and `test9_1_portfolio.csv`.
- **10.1–10.4:** Equal and unequal risk budgets, then maximum Sharpe with
  two sets of bounds. Risk parity uses `test5_2.csv`. Maximum Sharpe uses
  `test5_3.csv`, `test10_3_means.csv`, and risk-free rate 0.04.
- **11.1–11.2:** Asset and factor attribution for return and volatility.
  Inputs: the `test11_1_*.csv` and `test11_2_*.csv` files.
- **12.1–12.3:** European options and Greeks, American options with continuous
  dividends and Greeks, then American option values with cash dividends.
  Inputs: `test12_1.csv` for 12.1 and 12.2, `test12_3.csv` for 12.3.
- **13.1–13.9:** Kendall correlation, multivariate t, Gaussian and t copula
  fits, information criteria, tail dependence, and portfolio VaR/ES.
  Inputs: `test13_returns.csv` and, for 13.9, `test13_portfolio.csv`.

The reusable functions are in [risk_models.py](risk_models.py),
[portfolio_models.py](portfolio_models.py), and [option_models.py](option_models.py).
They can be imported without reading files or running any test.

## Course conventions and comparisons

The formulas follow [test_setup.jl](../test_setup.jl) and the course library,
especially `RiskStats.jl`, `fitted_model.jl`, `optimizers.jl`,
`expost_factor.jl`, `gbsm.jl`, `bt_american.jl`, `multivariate_t.jl`, and
`copula.jl`. The statistical inference notes support the marginal
maximum-likelihood fits. Test 13 profiles inverse degrees of freedom on
the same two grids as the course and uses one common chi-square draw per
scenario for the t copula.

- Actual reference names use underscores, such as `testout8_1.csv`.
  As in the generator, 8.6 measures ES and 10.3–10.4 use `test5_3.csv`.
- The 12.2 output follows the tree's derivative conventions. Rho varies the
  discount rate at fixed carry, and Theta differentiates with respect to
  time remaining. In 12.1, rho holds dividend yield fixed and theta measures
  elapsed time.
- 21 deterministic cases use `1e-8 + 1e-6 × abs(expected)` per numeric cell.
  Test 10.3 includes near-zero negative weights from the reference solver's
  boundary tolerance. Python retains the stated nonnegative bounds.
- 8.3, 8.6, 9.1, and 13.9 use independent Monte Carlo simulations with fixed
  Python seeds. Their reference CSVs are Julia simulations. The comparison
  allows five estimated standard errors of the difference (the Python
  standard error times sqrt(2)). This checks statistical agreement, not
  identical random draws, and is a local tolerance.
