"""Simple Python versions of the course functions used in Tests 1.1–7.6."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def missing_covariance(data, skip_missing=True, correlation=False):
    """Drop incomplete rows, or use the available rows for each pair."""
    data = pd.DataFrame(data)
    if skip_missing:
        data = data.dropna()
    return data.corr().to_numpy() if correlation else data.cov().to_numpy()


def ew_covariance(data, decay):
    """The last row is newest and receives the largest weight."""
    values = np.asarray(data, dtype=float)
    weights = (1 - decay) * decay ** np.arange(len(values) - 1, -1, -1)
    weights = weights / weights.sum()
    weighted_mean = weights @ values
    centered = values - weighted_mean
    return centered.T @ (weights[:, None] * centered)


def covariance_to_correlation(covariance):
    standard_deviation = np.sqrt(np.diag(covariance))
    return covariance / np.outer(standard_deviation, standard_deviation)


def near_psd(covariance):
    """Remove negative eigenvalues and keep the original diagonal."""
    standard_deviation = np.sqrt(np.diag(covariance))
    correlation = covariance_to_correlation(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    repaired = covariance_to_correlation(repaired)
    return repaired * np.outer(standard_deviation, standard_deviation)


def higham_psd(covariance, tolerance=1e-9, max_iterations=100):
    """Higham's alternating projections, using the course stopping rule."""
    standard_deviation = np.sqrt(np.diag(covariance))
    original = covariance_to_correlation(covariance)
    correlation = original.copy()
    correction = np.zeros_like(correlation)
    previous_distance = np.inf

    for _ in range(max_iterations):
        residual = correlation - correction
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        correction = projected - residual
        correlation = projected.copy()
        np.fill_diagonal(correlation, 1)

        distance = np.sum((correlation - original) ** 2)
        smallest_eigenvalue = np.linalg.eigvalsh(correlation).min()
        if (abs(distance - previous_distance) < tolerance
                and smallest_eigenvalue > -tolerance):
            return correlation * np.outer(standard_deviation, standard_deviation)
        previous_distance = distance

    raise RuntimeError("Higham's method did not converge.")


def chol_psd(covariance):
    """Cholesky factorization that also accepts zero pivots (PSD input)."""
    size = len(covariance)
    root = np.zeros((size, size))
    for column in range(size):
        diagonal = covariance[column, column] - root[column, :column] @ root[column, :column]
        if -1e-8 <= diagonal <= 0:
            diagonal = 0.0  # Remove tiny negative roundoff, as in the Julia example.
        if diagonal < 0:
            raise ValueError("The covariance matrix is not positive semidefinite.")
        root[column, column] = np.sqrt(diagonal)
        if root[column, column] == 0:
            continue
        for row in range(column + 1, size):
            previous = root[row, :column] @ root[column, :column]
            root[row, column] = (covariance[row, column] - previous) / root[column, column]
    return root


def simulate_normal(covariance, sample_count=100_000, repair=near_psd, seed=1234):
    """Generate zero-mean normal samples, repairing a non-PSD input if needed."""
    try:
        root = chol_psd(covariance)
    except ValueError:
        root = chol_psd(repair(covariance))
    random = np.random.default_rng(seed)
    standard_normals = random.standard_normal((sample_count, len(covariance)))
    return standard_normals @ root.T


def pca_root(covariance, explained_variance=0.99):
    """Keep the fewest positive-eigenvalue factors reaching the requested share."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    positive_count = np.count_nonzero(eigenvalues >= 1e-8)
    cumulative_share = np.cumsum(eigenvalues[:positive_count]) / eigenvalues.sum()
    factor_count = min(np.searchsorted(cumulative_share, explained_variance) + 1, positive_count)
    return eigenvectors[:, :factor_count] * np.sqrt(eigenvalues[:factor_count])


def simulate_pca(covariance, sample_count=100_000, explained_variance=0.99, seed=1234):
    root = pca_root(covariance, explained_variance)
    random = np.random.default_rng(seed)
    standard_normals = random.standard_normal((sample_count, root.shape[1]))
    return standard_normals @ root.T


def calculate_returns(prices, log_returns=False):
    values = prices.drop(columns="Date")
    price_ratio = values / values.shift(1)
    returns = np.log(price_ratio) if log_returns else price_ratio - 1
    returns.insert(0, "Date", prices["Date"])
    return returns.iloc[1:].reset_index(drop=True)


def fit_normal(values):
    # The course uses sample standard deviation (n - 1), not the MLE (n).
    return {"mu": np.mean(values), "sigma": np.std(values, ddof=1)}


def fit_t(values):
    # Tighten the default stopping rule so the fitted parameters match closely.
    def optimizer(objective, start, args=(), disp=0):
        fit = minimize(
            objective, start, args=args, method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-10},
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        return fit.x

    nu, mu, sigma = stats.t.fit(values, optimizer=optimizer)
    return {"mu": mu, "sigma": sigma, "nu": nu}


def fit_t_regression(predictors, response):
    """Fit intercept, slopes, and Student-t errors together by maximum likelihood."""
    predictors = np.asarray(predictors, dtype=float)
    response = np.asarray(response, dtype=float)
    predictor_scale = predictors.std(axis=0, ddof=1)
    response_scale = response.std(ddof=1)

    # Put the parameters on similar scales to help the optimizer.
    design = np.column_stack([np.ones(len(response)), predictors / predictor_scale])
    target = response / response_scale
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residuals = target - design @ coefficients
    start = np.r_[np.log(residuals.std(ddof=1)), np.log(8.0), coefficients]

    def negative_log_likelihood(parameters):
        sigma = np.exp(parameters[0])
        nu = 2 + np.exp(parameters[1])  # Positive scale and nu > 2, as in the course.
        residuals = target - design @ parameters[2:]
        return -stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma).sum()

    fit = minimize(
        negative_log_likelihood, start, method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-10},
    )
    if not fit.success:
        raise RuntimeError(fit.message)

    coefficients = fit.x[2:] * response_scale / np.r_[1, predictor_scale]
    result = {"mu": 0.0, "sigma": np.exp(fit.x[0]) * response_scale,
              "nu": 2 + np.exp(fit.x[1]), "Alpha": coefficients[0]}
    result.update({f"B{index}": value for index, value in enumerate(coefficients[1:], 1)})
    return result


def t_aicc(values, parameters):
    log_likelihood = stats.t.logpdf(
        values, df=parameters["nu"], loc=parameters["mu"], scale=parameters["sigma"],
    ).sum()
    parameter_count = 3  # mu, sigma, nu
    sample_count = len(values)
    return (-2 * log_likelihood + 2 * parameter_count
            + 2 * parameter_count * (parameter_count + 1) / (sample_count - parameter_count - 1))


def fit_nig_moments(values):
    """Invert the NIG moments, following library/fitted_model.jl."""
    mean = np.mean(values)
    variance = np.var(values, ddof=1)
    skewness = stats.skew(values, bias=True)
    excess_kurtosis = stats.kurtosis(values, fisher=True, bias=True)
    if excess_kurtosis <= (5 / 3) * skewness ** 2:
        raise ValueError("These sample moments are outside the NIG region.")

    ratio = skewness ** 2 / excess_kurtosis
    rho_squared = ratio / (3 - 4 * ratio)
    rho = np.sign(skewness) * np.sqrt(rho_squared)
    delta_gamma = 3 * (1 + 4 * rho_squared) / excess_kurtosis
    alpha = np.sqrt(delta_gamma / (variance * (1 - rho_squared) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho_squared)
    delta = delta_gamma / gamma
    mu = mean - delta * beta / gamma
    return {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}


def fit_nig_mle(values):
    a, b, mu, delta = stats.norminvgauss.fit(values)
    # SciPy uses a = alpha * delta and b = beta * delta.
    return {"mu": mu, "alpha": a / delta, "beta": b / delta, "delta": delta}
