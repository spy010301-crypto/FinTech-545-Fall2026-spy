"""Problem 5: keep residual correlation in market-model simulation."""

import numpy as np
import pandas as pd
from scipy import stats

from helpers import FOLDER, historical_risk, save_results


def main():
    prices = pd.read_csv(FOLDER / "problem5.csv").drop(columns="Day")
    returns = prices.pct_change().dropna()
    market = returns.MKT.to_numpy()
    stocks = returns[["A", "B"]].to_numpy()
    design = np.column_stack([np.ones(len(market)), market])
    coefficients = np.linalg.lstsq(design, stocks, rcond=None)[0]
    alphas, betas = coefficients
    residuals = stocks - design @ coefficients
    residual_covariance = np.cov(residuals, rowvar=False, ddof=1)
    independent_covariance = np.diag(np.diag(residual_covariance))
    market_variance = market.var(ddof=1)
    stock_covariance = np.cov(stocks, rowvar=False, ddof=1)
    portfolios = {"P1": np.array([1_000_000, 1_000_000]),
                  "P2": np.array([1_000_000, -1_000_000])}
    results = {"regressions": {}, "residual_correlation": float(np.corrcoef(residuals.T)[0, 1]),
               "residual_covariance": residual_covariance.tolist(),
               "stock_covariance": stock_covariance.tolist(), "VaR_dollars": {}}
    for column, asset in enumerate(["A", "B"]):
        results["regressions"][asset] = {
            "alpha": float(alphas[column]), "beta": float(betas[column]),
            "residual_standard_deviation": float(residuals[:, column].std(ddof=1)),
        }

    random = np.random.default_rng(1234)
    draws = random.standard_normal((100_000, 3))
    simulated_market = draws[:, 0] * np.sqrt(market_variance)
    for name, covariance in [("full_residual", residual_covariance),
                             ("independent_residual", independent_covariance)]:
        errors = draws[:, 1:] @ np.linalg.cholesky(covariance).T
        # The assignment assumes zero expected returns: do not add the fitted alphas.
        simulated_stocks = simulated_market[:, None] * betas + errors
        results["VaR_dollars"][name] = {
            portfolio: historical_risk(simulated_stocks @ weights)["VaR"]
            for portfolio, weights in portfolios.items()}

    results["VaR_dollars"]["delta_normal"] = {
        portfolio: float(-stats.norm.ppf(0.05) * np.sqrt(weights @ stock_covariance @ weights))
        for portfolio, weights in portfolios.items()}
    model_covariance = market_variance * np.outer(betas, betas) + residual_covariance
    shortcut_covariance = market_variance * np.outer(betas, betas) + independent_covariance
    results["full_model_covariance"] = model_covariance.tolist()
    results["maximum_covariance_reconstruction_error"] = float(np.max(np.abs(model_covariance - stock_covariance)))
    results["shortcut_analytic_VaR"] = {
        portfolio: float(-stats.norm.ppf(0.05) * np.sqrt(weights @ shortcut_covariance @ weights))
        for portfolio, weights in portfolios.items()}
    results["relative_shortcut_VaR_change"] = {
        portfolio: float(results["shortcut_analytic_VaR"][portfolio]
                         / results["VaR_dollars"]["delta_normal"][portfolio] - 1)
        for portfolio in portfolios}
    save_results(5, results)


if __name__ == "__main__":
    main()
