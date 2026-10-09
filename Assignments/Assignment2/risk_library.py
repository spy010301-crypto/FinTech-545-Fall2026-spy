"""Reusable Python risk calculations, translated from the Week 1-5 examples.

Functions accept data and return results. They never read CSVs, print answers,
or write files. Matrix inputs are NumPy arrays: rows are days, columns are assets.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def arithmetic_returns(prices):
    """Return current/previous - 1 for a pandas price Series or DataFrame."""
    return prices.pct_change().dropna()


def moments(values):
    """Sample variance; standardized central moments, as in Week 1."""
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)),
        "skewness": float(stats.skew(values, bias=True)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=True)),
    }


def aicc(log_likelihood, parameters, observations):
    """Corrected AIC; parameters counts the model's free parameters."""
    return (-2 * log_likelihood + 2 * parameters
            + 2 * parameters * (parameters + 1) / (observations - parameters - 1))


def exponential_weights(observations, decay):
    """Normalized finite weights, ordered oldest to newest."""
    weights = decay ** np.arange(observations - 1, -1, -1)
    return weights / weights.sum()


def ew_volatility(values, weights):
    """Weighted SD using normalized weights and a weighted mean (Week 3)."""
    weighted_mean = weights @ values
    return np.sqrt(weights @ (values - weighted_mean) ** 2)


def correlation_to_covariance(correlation, standard_deviations):
    """Covariance = correlation times each pair's two standard deviations."""
    return correlation * np.outer(standard_deviations, standard_deviations)


def portfolio_variance(covariance, weights):
    """Return w'Cov w; weights and covariance must use the same asset order."""
    return float(weights @ covariance @ weights)


def delta_normal_var(covariance, weights, alpha=0.05):
    """Zero-mean normal VaR; dollar weights produce a dollar VaR."""
    return float(-stats.norm.ppf(alpha) * np.sqrt(portfolio_variance(covariance, weights)))


def normal_var(pnl, alpha=0.05):
    """Normal VaR from a P&L Series/array's actual mean and sample SD."""
    return float(-stats.norm.ppf(alpha, loc=pnl.mean(), scale=pnl.std(ddof=1)))


def historical_risk(pnl, alpha=0.05):
    """Course VaR/ES: average floor/ceiling order statistics, 1-based ranks."""
    ordered = np.sort(pnl)
    rank = alpha * len(ordered)
    lower = max(int(np.floor(rank)) - 1, 0)
    upper = max(int(np.ceil(rank)) - 1, 0)
    cutoff = (ordered[lower] + ordered[upper]) / 2
    # Include cutoff ties in ES; keep negative VaR when the tail is still a gain.
    return {"VaR": float(-cutoff), "ES": float(-ordered[ordered <= cutoff].mean())}


def near_psd(correlation):
    """Rebonato-Jackel: clip negative eigenvalues, then restore a unit diagonal."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    repaired = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def higham(correlation, tolerance=1e-12, max_iterations=1000):
    """Alternate PSD/unit-diagonal projections with Dykstra correction."""
    repaired = correlation.copy()
    correction = np.zeros_like(repaired)
    for _ in range(max_iterations):
        previous = repaired.copy()
        residual = repaired - correction
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        correction = projected - residual
        repaired = projected.copy()
        np.fill_diagonal(repaired, 1)
        if (np.linalg.norm(repaired - previous, "fro") < tolerance
                and np.linalg.eigvalsh(repaired).min() > -tolerance):
            break
    return repaired


def fit_margin(values):
    """Fit normal/t margins by MLE; return the AICc winner and fit details."""
    location, scale = stats.norm.fit(values)
    normal = stats.norm(loc=location, scale=scale)
    nu, t_location, t_scale = stats.t.fit(values)
    student_t = stats.t(df=nu, loc=t_location, scale=t_scale)
    # MLE normal scale uses n, not the descriptive sample SD's n-1.
    normal_aicc = aicc(float(normal.logpdf(values).sum()), 2, len(values))
    t_aicc = aicc(float(student_t.logpdf(values).sum()), 3, len(values))
    chosen = normal if normal_aicc < t_aicc else student_t
    details = {
        "normal": {"mu": float(location), "sigma": float(scale), "AICc": float(normal_aicc)},
        "t": {"mu": float(t_location), "sigma_scale": float(t_scale),
              "nu": float(nu), "AICc": float(t_aicc)},
        "chosen": "normal" if normal_aicc < t_aicc else "t",
    }
    return chosen, details


def copula_correlation(uniforms):
    """Return Kendall tau and R = sin(pi*tau/2), repairing R if needed."""
    assets = uniforms.shape[1]
    tau = np.eye(assets)
    for first in range(assets):
        for second in range(first + 1, assets):
            tau[first, second] = tau[second, first] = stats.kendalltau(
                uniforms[:, first], uniforms[:, second]).statistic
    correlation = np.sin(np.pi * tau / 2)
    if np.linalg.eigvalsh(correlation).min() < 0:
        correlation = higham(correlation)
    return tau, correlation


def copula_log_density(uniforms, correlation, nu=None):
    """Daily joint log density minus marginal log densities; None means Gaussian."""
    assets = uniforms.shape[1]
    if nu is None:
        scores = stats.norm.ppf(uniforms)
        joint = stats.multivariate_normal.logpdf(scores, mean=np.zeros(assets), cov=correlation)
        return joint - stats.norm.logpdf(scores).sum(axis=1)
    scores = stats.t.ppf(uniforms, df=nu)
    joint = stats.multivariate_t.logpdf(scores, loc=np.zeros(assets), shape=correlation, df=nu)
    return joint - stats.t.logpdf(scores, df=nu).sum(axis=1)


def fit_t_copula(uniforms, correlation, min_nu=2.0001, max_nu=100):
    """Fit nu with fixed R: profile 1/nu, then refine the best grid interval."""
    grid = np.linspace(1 / max_nu, 1 / min_nu, 80)
    grid_ll = [copula_log_density(uniforms, correlation, nu=1 / value).sum() for value in grid]
    best = int(np.argmax(grid_ll))
    fit = minimize_scalar(
        lambda value: -copula_log_density(uniforms, correlation, nu=1 / value).sum(),
        bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]),
        method="bounded", options={"xatol": 1e-10},
    )
    # Keep a boundary maximum if it beats the refined interior solution.
    candidates = [float(grid[best]), float(fit.x)]
    inverse_nu = max(candidates, key=lambda value: copula_log_density(
        uniforms, correlation, nu=1 / value).sum())
    return 1 / inverse_nu


def simulate_copulas(correlation, nu, simulations=100_000, seed=1234):
    """Return Gaussian/t uniform draws, using the same underlying normal draws."""
    random = np.random.default_rng(seed)
    assets = len(correlation)
    normal_scores = random.standard_normal((simulations, assets)) @ np.linalg.cholesky(correlation).T
    gaussian_uniforms = stats.norm.cdf(normal_scores)
    # One shared chi-square draw per day creates t-copula tail dependence.
    t_scores = normal_scores * np.sqrt(nu / random.chisquare(nu, simulations))[:, None]
    return gaussian_uniforms, stats.t.cdf(t_scores, df=nu)


def joint_tail_counts(uniforms, asset_names, alpha=0.025, days=1000):
    """All pairs' lower/upper joint-tail frequencies, scaled to days."""
    counts = {}
    for first in range(uniforms.shape[1]):
        for second in range(first + 1, uniforms.shape[1]):
            counts[f"{asset_names[first]}/{asset_names[second]}"] = {
                "lower": float(days * np.mean((uniforms[:, first] < alpha) & (uniforms[:, second] < alpha))),
                "upper": float(days * np.mean((uniforms[:, first] > 1 - alpha) & (uniforms[:, second] > 1 - alpha))),
            }
    return counts


def copula_tail_dependence(rho, nu=None):
    """Gaussian tail dependence is zero for |rho|<1; t uses the Week 5 formula."""
    if nu is None:
        return 0.0
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1))


def fit_market_model(market, stocks):
    """OLS with an intercept: return alphas, betas, and the residual matrix."""
    design = np.column_stack([np.ones(len(market)), market])
    coefficients = np.linalg.lstsq(design, stocks, rcond=None)[0]
    alphas, betas = coefficients
    residuals = stocks - design @ coefficients
    return alphas, betas, residuals


def simulate_market_model(betas, market_variance, residual_covariance, draws):
    """Zero-mean stock returns; draws has market first, then one column per stock."""
    simulated_market = draws[:, 0] * np.sqrt(market_variance)
    errors = draws[:, 1:] @ np.linalg.cholesky(residual_covariance).T
    # Omit fitted alphas to honor the zero-expected-return assumption.
    return simulated_market[:, None] * betas + errors
