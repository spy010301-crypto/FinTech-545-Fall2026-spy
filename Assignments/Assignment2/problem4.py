"""Problem 4: choose margins, compare copulas, and simulate portfolio risk."""

import numpy as np
import pandas as pd

from helpers import FOLDER, save_results
from risk_library import (
    aicc, copula_correlation, copula_log_density, copula_tail_dependence,
    fit_margin, fit_t_copula, historical_risk, joint_tail_counts, moments,
    simulate_copulas,
)


def main():
    returns = pd.read_csv(FOLDER / "problem4.csv")
    observations = len(returns)
    ranks = (returns.rank() / (observations + 1)).to_numpy()
    uniforms = np.zeros_like(ranks)
    margins = []
    results = {"moments": {}, "outlier_sensitivity": {}, "margins": {}}

    for column, asset in enumerate(returns.columns):
        values = returns[asset]
        results["moments"][asset] = moments(values)
        outlier = (values - values.mean()).abs().idxmax()
        trimmed = values.drop(outlier)
        results["outlier_sensitivity"][asset] = {
            "row_1based": int(outlier + 1), "return": float(values.iloc[outlier]),
            "moments_without_one_observation": moments(trimmed),
        }
        chosen, details = fit_margin(values)
        margins.append(chosen)
        uniforms[:, column] = chosen.cdf(values)
        results["margins"][asset] = details

    uniforms = np.clip(uniforms, 1e-12, 1 - 1e-12)  # Only guard numerical ppf endpoints.
    tau, correlation = copula_correlation(uniforms)
    results["kendall_tau"] = tau.tolist()
    results["copula_correlation"] = correlation.tolist()

    gaussian_daily_ll = copula_log_density(uniforms, correlation)
    copula_nu = fit_t_copula(uniforms, correlation)
    t_daily_ll = copula_log_density(uniforms, correlation, nu=copula_nu)
    results["copulas"] = {}
    for name, daily_ll, parameters in [("Gaussian", gaussian_daily_ll, 0), ("t", t_daily_ll, 1)]:
        log_likelihood = float(daily_ll.sum())
        # Week 5 freezes the common Kendall R: only t's nu is a new free parameter.
        results["copulas"][name] = {
            "log_likelihood": log_likelihood, "free_parameters": parameters,
            "nu": None if name == "Gaussian" else float(copula_nu),
            "AICc": float(aicc(log_likelihood, parameters, observations)),
            "BIC": float(-2 * log_likelihood + parameters * np.log(observations)),
        }

    gaussian_uniforms, t_uniforms = simulate_copulas(correlation, copula_nu, simulations=100_000, seed=1234)
    historical_pnl = 1_000_000 * returns.sum(axis=1).to_numpy()
    results["portfolio_risk"] = {"historical": {
        "5_percent": historical_risk(historical_pnl, 0.05),
        "1_percent": historical_risk(historical_pnl, 0.01),
    }}
    asset_names = list(returns.columns)
    results["joint_tail_days_per_1000"] = {"observed_ranks": joint_tail_counts(ranks, asset_names)}
    for name, simulated_uniforms in [("Gaussian", gaussian_uniforms), ("t", t_uniforms)]:
        simulated_returns = np.column_stack([
            margin.ppf(simulated_uniforms[:, column]) for column, margin in enumerate(margins)])
        pnl = 1_000_000 * simulated_returns.sum(axis=1)
        results["portfolio_risk"][name] = {"5_percent": historical_risk(pnl, 0.05),
                                           "1_percent": historical_risk(pnl, 0.01)}
        results["joint_tail_days_per_1000"][name] = joint_tail_counts(simulated_uniforms, asset_names)

    difference = t_daily_ll - gaussian_daily_ll
    # Define "own outer 5%" with observed ranks, consistently with part (b).
    middle = ((ranks >= 0.05) & (ranks <= 0.95)).all(axis=1)
    fitted_middle = ((uniforms >= 0.05) & (uniforms <= 0.95)).all(axis=1)
    results["likelihood_gain"] = {
        "total": float(difference.sum()), "middle_days": int(middle.sum()),
        "middle_gain": float(difference[middle].sum()),
        "middle_share": float(difference[middle].sum() / difference.sum()),
        "outside_middle_gain": float(difference[~middle].sum()),
        "alternative_fitted_CDF_middle_days": int(fitted_middle.sum()),
        "alternative_fitted_CDF_middle_gain": float(difference[fitted_middle].sum()),
    }
    # Also save every day's contribution, so part (i) is directly inspectable.
    output = FOLDER / "results"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"row_1based": np.arange(1, observations + 1),
                  "gaussian_log_density": gaussian_daily_ll, "t_log_density": t_daily_ll,
                  "t_minus_gaussian": difference, "all_empirical_ranks_in_middle_90_percent": middle,
                  "U_X1": uniforms[:, 0], "U_X2": uniforms[:, 1], "U_X3": uniforms[:, 2]}).to_csv(
                      output / "problem4_daily_likelihood.csv", index=False)
    first, second = max([(0, 1), (0, 2), (1, 2)], key=lambda pair: correlation[pair])
    rho = correlation[first, second]
    tail_dependence = copula_tail_dependence(rho, copula_nu)
    results["tail_dependence"] = {"pair": f"X{first + 1}/X{second + 1}", "rho": float(rho),
                                  "t": float(tail_dependence), "Gaussian": copula_tail_dependence(rho)}
    save_results(4, results)


if __name__ == "__main__":
    main()
