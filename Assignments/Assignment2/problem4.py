"""Problem 4: choose margins, compare copulas, and simulate portfolio risk."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from helpers import FOLDER, aicc, higham, historical_risk, moments, save_results


def copula_log_density(uniforms, correlation, nu=None):
    """Joint log density minus marginal log densities (Week 5)."""
    if nu is None:
        scores = stats.norm.ppf(uniforms)
        joint = stats.multivariate_normal.logpdf(scores, mean=np.zeros(3), cov=correlation)
        return joint - stats.norm.logpdf(scores).sum(axis=1)
    scores = stats.t.ppf(uniforms, df=nu)
    joint = stats.multivariate_t.logpdf(scores, loc=np.zeros(3), shape=correlation, df=nu)
    return joint - stats.t.logpdf(scores, df=nu).sum(axis=1)


def joint_tail_counts(uniforms):
    """Joint 2.5% tail frequencies, expressed as expected days per 1,000."""
    counts = {}
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        counts[f"X{first + 1}/X{second + 1}"] = {
            "lower": float(1000 * np.mean((uniforms[:, first] < 0.025) & (uniforms[:, second] < 0.025))),
            "upper": float(1000 * np.mean((uniforms[:, first] > 0.975) & (uniforms[:, second] > 0.975))),
        }
    return counts


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
        # Use actual MLE parameters for this likelihood-based margin comparison.
        # Normal MLE scale uses n; the descriptive sample variance above uses n-1.
        location, scale = stats.norm.fit(values)
        normal = stats.norm(loc=location, scale=scale)
        nu, t_location, t_scale = stats.t.fit(values)
        student_t = stats.t(df=nu, loc=t_location, scale=t_scale)
        normal_ll = float(normal.logpdf(values).sum())
        t_ll = float(student_t.logpdf(values).sum())
        normal_aicc = aicc(normal_ll, 2, observations)
        t_aicc = aicc(t_ll, 3, observations)
        chosen = normal if normal_aicc < t_aicc else student_t
        margins.append(chosen)
        uniforms[:, column] = chosen.cdf(values)
        results["margins"][asset] = {
            "normal": {"mu": float(location), "sigma": float(scale), "AICc": float(normal_aicc)},
            "t": {"mu": float(t_location), "sigma_scale": float(t_scale),
                  "nu": float(nu), "AICc": float(t_aicc)},
            "chosen": "normal" if normal_aicc < t_aicc else "t",
        }

    uniforms = np.clip(uniforms, 1e-12, 1 - 1e-12)  # Only guard numerical ppf endpoints.
    tau = np.eye(3)
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        tau[first, second] = tau[second, first] = stats.kendalltau(uniforms[:, first], uniforms[:, second]).statistic
    correlation = np.sin(np.pi * tau / 2)
    if np.linalg.eigvalsh(correlation).min() < 0:
        correlation = higham(correlation)
    results["kendall_tau"] = tau.tolist()
    results["copula_correlation"] = correlation.tolist()

    gaussian_daily_ll = copula_log_density(uniforms, correlation)
    # Profile 1/nu on the course range 2.0001 <= nu <= 100, then refine the best grid interval.
    grid = np.linspace(1 / 100, 1 / 2.0001, 80)
    grid_ll = [copula_log_density(uniforms, correlation, nu=1 / value).sum() for value in grid]
    best = int(np.argmax(grid_ll))
    fit = minimize_scalar(
        lambda value: -copula_log_density(uniforms, correlation, nu=1 / value).sum(),
        bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]),
        method="bounded", options={"xatol": 1e-10},
    )
    candidates = [float(grid[best]), float(fit.x)]  # Keep a boundary maximum if it wins.
    inverse_nu = max(candidates, key=lambda value: copula_log_density(uniforms, correlation, nu=1 / value).sum())
    copula_nu = 1 / inverse_nu
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

    random = np.random.default_rng(1234)
    simulations = 100_000
    normal_scores = random.standard_normal((simulations, 3)) @ np.linalg.cholesky(correlation).T
    gaussian_uniforms = stats.norm.cdf(normal_scores)
    # One shared chi-square draw per day creates t-copula tail dependence.
    t_scores = normal_scores * np.sqrt(copula_nu / random.chisquare(copula_nu, simulations))[:, None]
    t_uniforms = stats.t.cdf(t_scores, df=copula_nu)
    historical_pnl = 1_000_000 * returns.sum(axis=1).to_numpy()
    results["portfolio_risk"] = {"historical": {
        "5_percent": historical_risk(historical_pnl, 0.05),
        "1_percent": historical_risk(historical_pnl, 0.01),
    }}
    results["joint_tail_days_per_1000"] = {"observed_ranks": joint_tail_counts(ranks)}
    for name, simulated_uniforms in [("Gaussian", gaussian_uniforms), ("t", t_uniforms)]:
        simulated_returns = np.column_stack([
            margin.ppf(simulated_uniforms[:, column]) for column, margin in enumerate(margins)])
        pnl = 1_000_000 * simulated_returns.sum(axis=1)
        results["portfolio_risk"][name] = {"5_percent": historical_risk(pnl, 0.05),
                                           "1_percent": historical_risk(pnl, 0.01)}
        results["joint_tail_days_per_1000"][name] = joint_tail_counts(simulated_uniforms)

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
    tail_dependence = 2 * stats.t.cdf(-np.sqrt((copula_nu + 1) * (1 - rho) / (1 + rho)), df=copula_nu + 1)
    results["tail_dependence"] = {"pair": f"X{first + 1}/X{second + 1}", "rho": float(rho),
                                  "t": float(tail_dependence), "Gaussian": 0.0}
    save_results(4, results)


if __name__ == "__main__":
    main()
