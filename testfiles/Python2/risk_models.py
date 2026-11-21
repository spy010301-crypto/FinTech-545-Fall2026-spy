"""Distribution fits, tail risk, and copulas for course tests 8, 9, and 13."""

import numpy as np
from scipy import optimize, special, stats


def fit_normal(returns):
    return stats.norm(loc=np.mean(returns), scale=np.std(returns, ddof=1))


def fit_student(returns):
    """Fit location, scale, and degrees of freedom by maximum likelihood."""
    center, unit = np.mean(returns), np.std(returns, ddof=1)
    standardized = (returns - center) / unit

    def objective(parameters):
        location, log_scale, degrees = parameters
        residual = (standardized - location) / np.exp(log_scale)
        squared = residual**2
        log_term = np.log1p(squared / degrees)
        likelihood = stats.t.logpdf(residual, degrees) - log_scale
        # Analytic score avoids finite difference noise in the degrees of freedom.
        location_score = (degrees + 1) * residual / (degrees + squared) / np.exp(log_scale)
        scale_score = -1 + (degrees + 1) * squared / (degrees + squared)
        degrees_score = 0.5 * (
            special.digamma((degrees + 1) / 2) - special.digamma(degrees / 2)
            - 1 / degrees - log_term
            + (degrees + 1) * squared / (degrees * (degrees + squared))
        )
        score = np.array([location_score.mean(), scale_score.mean(), degrees_score.mean()])
        return -likelihood.mean(), -score

    fit = optimize.minimize(
        objective, [0, 0, 8], jac=True, method="L-BFGS-B",
        bounds=[(None, None), (np.log(1e-6 / unit), None), (2.0001, None)],
        options={"gtol": 1e-11, "ftol": 1e-15},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    location, log_scale, degrees = fit.x
    return stats.t(degrees, loc=center + unit * location, scale=unit * np.exp(log_scale))


def distribution_risk(distribution, measure, alpha=0.05):
    cutoff = distribution.ppf(alpha)
    if measure == "VaR":
        return -cutoff
    # Integrate the same finite tail as the course's RiskStats.jl.
    tail_start = distribution.ppf(1e-12)
    return -distribution.expect(lambda value: value, lb=tail_start, ub=cutoff) / alpha


def sample_risk(pnl, measure, alpha=0.05):
    ordered = np.sort(pnl)
    rank = len(ordered) * alpha
    # Julia uses 1-based ranks, not NumPy's default percentile interpolation.
    cutoff = (ordered[max(0, int(np.floor(rank)) - 1)]
              + ordered[int(np.ceil(rank)) - 1]) / 2
    if measure == "VaR":
        return -cutoff
    return -ordered[ordered <= cutoff].mean()


def simulation_error(samples, measure, centered=False, alpha=0.05):
    """Estimated standard error of a simulated VaR or ES."""
    cutoff = -sample_risk(samples, "VaR", alpha)
    if measure == "VaR":
        low, high = np.quantile(samples, [alpha - 0.005, alpha + 0.005])
        # Local quantile slope is the reciprocal of density at the cutoff.
        influence = ((samples <= cutoff) - alpha) * (high - low) / 0.01
    else:
        influence = np.maximum(cutoff - samples, 0) / alpha
    if centered:
        influence = influence + samples
    return influence.std(ddof=1) / np.sqrt(len(samples))


def kendall_correlation(observations):
    count = observations.shape[1]
    correlation = np.eye(count)
    for first in range(count):
        for second in range(first + 1, count):
            tau = stats.kendalltau(observations[:, first], observations[:, second]).statistic
            correlation[first, second] = correlation[second, first] = np.sin(np.pi * tau / 2)

    if np.linalg.eigvalsh(correlation).min() < -1e-8:
        # Higham's alternating projections: PSD cone and unit diagonal.
        correction = np.zeros_like(correlation)
        for _ in range(1000):
            residual = correlation - correction
            eigenvalues, eigenvectors = np.linalg.eigh(residual)
            projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
            correction = projected - residual
            correlation = (projected + projected.T) / 2
            np.fill_diagonal(correlation, 1)
            if np.linalg.eigvalsh(correlation).min() >= -1e-10:
                break
        else:
            raise RuntimeError("Correlation repair did not converge")
    smallest = np.linalg.eigvalsh(correlation).min()
    if smallest < 1e-8:
        ridge = 1e-8 - smallest
        correlation = (correlation + ridge * np.eye(count)) / (1 + ridge)
    return correlation


def profile_degrees(log_likelihood):
    """The course searches 1/nu on a coarse grid, then refines it locally."""
    grid = np.linspace(0.01, 0.49, 200)
    likelihoods = [log_likelihood(1 / inverse_degrees) for inverse_degrees in grid]
    best = int(np.argmax(likelihoods))
    fine = np.linspace(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)], 200)
    likelihoods = [log_likelihood(1 / inverse_degrees) for inverse_degrees in fine]
    best = int(np.argmax(likelihoods))
    return 1 / fine[best], likelihoods[best]


def fit_multivariate_student(returns, correlation):
    location = returns.mean(axis=0)
    sd = returns.std(axis=0, ddof=1)
    covariance = correlation * np.outer(sd, sd)

    def likelihood(degrees):
        scale = covariance * (degrees - 2) / degrees
        return stats.multivariate_t.logpdf(returns, loc=location, shape=scale, df=degrees).sum()

    degrees, log_likelihood = profile_degrees(likelihood)
    scale = covariance * (degrees - 2) / degrees
    return location, scale, degrees, log_likelihood


def copula_log_likelihood(uniforms, correlation, degrees=None):
    if degrees is None:
        scores = stats.norm.ppf(uniforms)
        joint = stats.multivariate_normal.logpdf(scores, cov=correlation)
        marginal = stats.norm.logpdf(scores)
    else:
        scores = stats.t.ppf(uniforms, degrees)
        joint = stats.multivariate_t.logpdf(scores, shape=correlation, df=degrees)
        marginal = stats.t.logpdf(scores, degrees)
    return joint.sum() - marginal.sum()


def simulate_portfolio(margins, correlation, exposures, count, seed, degrees=None):
    generator = np.random.default_rng(seed)
    scores = generator.standard_normal((count, len(margins))) @ np.linalg.cholesky(correlation).T
    if degrees is None:
        uniforms = stats.norm.cdf(scores)
    else:
        # Each scenario shares one chi-square shock across all assets.
        shock = np.sqrt(degrees / generator.chisquare(degrees, count))
        uniforms = stats.t.cdf(scores * shock[:, None], degrees)
    uniforms = np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps)
    returns = np.column_stack([margin.ppf(column) for margin, column in zip(margins, uniforms.T)])
    return returns * exposures
