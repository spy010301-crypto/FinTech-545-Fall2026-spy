"""Small reusable portfolio and regression functions for Problems 4 and 5.

The formulas follow Week 7's portfolio/CAPM examples and Week 9's OLS example.
Returns are decimals. Pass means, covariance, and risk-free rate at one frequency.
"""

import numpy as np
from scipy.optimize import minimize


def ols(returns, factors, factor_names):
    """OLS with an intercept, usual coefficient SEs, and Gaussian AICc.

    The residual variance used for coefficient SEs divides by n-p. The maximized
    Gaussian likelihood instead divides by n. AICc counts p coefficients AND the
    estimated error variance, so k=p+1. These are different purposes/denominators.
    """
    returns = np.asarray(returns, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if factors.ndim == 1:
        factors = factors[:, None]
    design = np.column_stack([np.ones(len(returns)), factors])
    coefficients = np.linalg.lstsq(design, returns, rcond=None)[0]
    residuals = returns - design @ coefficients
    n, p = design.shape
    squared_error = residuals @ residuals
    variance = squared_error / (n - p)
    standard_errors = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * variance)
    log_likelihood = -n / 2 * (np.log(2 * np.pi * squared_error / n) + 1)
    k = p + 1
    names = ["alpha"] + list(factor_names)
    return {
        "n": n,
        "coefficients": dict(zip(names, coefficients)),
        "standard_errors": dict(zip(names, standard_errors)),
        "t_statistics": dict(zip(names, coefficients / standard_errors)),
        "r_squared": 1 - squared_error / np.sum((returns - returns.mean()) ** 2),
        "log_likelihood": log_likelihood,
        "aicc_parameter_count": k,
        "aicc": -2 * log_likelihood + 2 * k + 2 * k * (k + 1) / (n - k - 1),
        "residuals": residuals,
    }


def maximum_sharpe(mean, covariance, risk_free, long_only=True, cap=None):
    """Fully invested tangency weights; cap optionally limits the last asset.

    Without bounds, normalize inverse-covariance times excess mean only when
    its sum is positive. A nonpositive sum means the positive-Sharpe supremum
    has no finite, fully invested tangency weights; normalizing a negative sum
    would instead return the negative-Sharpe minimum. With bounds, minimize
    negative Sharpe as in library/optimizers.jl::maxSR.
    """
    mean = np.asarray(mean)
    excess = mean - risk_free
    if not long_only:
        direction = np.linalg.solve(covariance, excess)
        normalizer = direction.sum()
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("No finite positive-Sharpe fully invested tangency portfolio")
        return direction / normalizer

    def objective(weights):
        volatility = np.sqrt(weights @ covariance @ weights)
        return -(weights @ excess) / volatility

    def gradient(weights):
        volatility = np.sqrt(weights @ covariance @ weights)
        return -excess / volatility + (weights @ excess) * (covariance @ weights) / volatility**3

    n = len(mean)
    bounds = [(0, 1)] * n
    start = np.full(n, 1 / n)
    if cap is not None:
        bounds[-1] = (0, cap)
        start[-1] = min(cap, 1 / n)
        start[:-1] = (1 - start[-1]) / (n - 1)
    result = minimize(
        objective, start, jac=gradient, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones(n)},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def efficient_frontier(mean, covariance, points=70):
    """Long-only minimum-variance portfolios, only the efficient upper branch."""
    n = len(mean)
    # Scaling the objective changes no optimum, but avoids tiny squared-return
    # objective values causing the numerical solver to stop prematurely.
    objective = lambda w: 10000 * (w @ covariance @ w)
    gradient = lambda w: 20000 * (covariance @ w)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1,
                    "jac": lambda w: np.ones(n)}]
    minimum = minimize(objective, np.full(n, 1 / n), jac=gradient, bounds=[(0, 1)] * n,
                       constraints=constraints, method="SLSQP", options={"ftol": 1e-12})
    if not minimum.success:
        raise RuntimeError(minimum.message)
    weights = minimum.x
    frontier = []
    for target in np.linspace(mean @ weights, mean.max(), points):
        target_constraint = {"type": "eq", "fun": lambda w: mean @ w - target,
                             "jac": lambda w: mean}
        fit = minimize(objective, weights, jac=gradient, bounds=[(0, 1)] * n,
                       constraints=constraints + [target_constraint], method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 500})
        if not fit.success:
            raise RuntimeError(fit.message)
        weights = fit.x
        frontier.append({"mean": mean @ weights,
                         "volatility": np.sqrt(weights @ covariance @ weights),
                         "weights": weights.copy()})
    return frontier


def fractional_expected_shortfall(returns, alpha):
    """Exact empirical ES: give the boundary observation fractional tail weight.

    At 1% of 240 observations this averages two full observations plus 0.4 of the
    third. This is a sensitivity alongside the course's two-observation estimate.
    """
    ordered = np.sort(returns)
    tail_count = alpha * len(ordered)
    whole = int(np.floor(tail_count))
    fraction = tail_count - whole
    tail_sum = ordered[:whole].sum()
    if fraction:
        tail_sum += fraction * ordered[whole]
    return -tail_sum / tail_count
