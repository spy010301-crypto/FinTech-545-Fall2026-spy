"""Small, shared calculations translated from the Week 1-5 examples."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

FOLDER = Path(__file__).resolve().parent


def moments(values):
    """Sample variance; standardized central moments, as in Week 1."""
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)),
        "skewness": float(stats.skew(values, bias=True)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=True)),
    }


def historical_risk(pnl, alpha=0.05):
    """Week 4: average the floor/ceiling order statistics, using 1-based ranks."""
    ordered = np.sort(pnl)
    rank = alpha * len(ordered)
    lower = max(int(np.floor(rank)) - 1, 0)
    upper = max(int(np.ceil(rank)) - 1, 0)
    cutoff = (ordered[lower] + ordered[upper]) / 2
    # Keep the sign: negative VaR means the tail still contains a gain.
    return {"VaR": float(-cutoff), "ES": float(-ordered[ordered <= cutoff].mean())}


def aicc(log_likelihood, parameters, observations):
    return (-2 * log_likelihood + 2 * parameters
            + 2 * parameters * (parameters + 1) / (observations - parameters - 1))


def near_psd(correlation):
    """Rebonato-Jackel: clip negative eigenvalues, then restore a unit diagonal."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    repaired = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def higham(correlation):
    """Week 3: alternate PSD and unit-diagonal projections with Dykstra correction."""
    repaired = correlation.copy()
    correction = np.zeros_like(repaired)
    for _ in range(1000):
        previous = repaired.copy()
        residual = repaired - correction
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        correction = projected - residual
        repaired = projected.copy()
        np.fill_diagonal(repaired, 1)
        if (np.linalg.norm(repaired - previous, "fro") < 1e-12
                and np.linalg.eigvalsh(repaired).min() > -1e-12):
            break
    return repaired


def save_results(problem, results):
    """Save the unrounded numbers as well as printing them."""
    output = FOLDER / "results"
    output.mkdir(exist_ok=True)
    text = json.dumps(results, indent=2, allow_nan=False)
    (output / f"problem{problem}_results.json").write_text(text + "\n")
    print(text)
