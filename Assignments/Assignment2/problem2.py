"""Problem 2: compare full-history and recent-regime volatility estimates."""

import numpy as np
import pandas as pd
from scipy import stats

from helpers import FOLDER, historical_risk, moments, save_results


def main():
    prices = pd.read_csv(FOLDER / "problem2.csv")
    raw_returns = prices.Price.pct_change().dropna().to_numpy()
    returns = raw_returns - raw_returns.mean()
    position = 1_000_000
    recent_days = 40  # Visual estimate recorded in Predictions.md, not fitted to VaR.
    results = {"removed_mean": float(raw_returns.mean()), "moments": moments(returns),
               "recent_days": recent_days,
               "earlier_standard_deviation": float(returns[:-recent_days].std(ddof=1)),
               "recent_standard_deviation": float(returns[-recent_days:].std(ddof=1))}
    standard_deviation = returns.std(ddof=1)
    var = {"normal_equal": float(-stats.norm.ppf(0.05) * standard_deviation * position)}

    results["EW"] = {}
    for decay in [0.97, 0.94]:
        # Newest return has exponent 0. Normalize the finite, 500-day weights.
        weights = decay ** np.arange(len(returns) - 1, -1, -1)
        weights = weights / weights.sum()
        weighted_mean = weights @ returns
        volatility = np.sqrt(weights @ (returns - weighted_mean) ** 2)
        effective_sample_size = 1 / (weights @ weights)
        volatility_se = volatility / np.sqrt(2 * effective_sample_size)
        var[f"normal_EW_{decay}"] = float(-stats.norm.ppf(0.05) * volatility * position)
        results["EW"][str(decay)] = {
            "volatility": float(volatility),
            "effective_sample_size": float(effective_sample_size),
            "half_life_days": float(np.log(0.5) / np.log(decay)),
            "recent_weight_share": float(weights[-recent_days:].sum()),
            "volatility_standard_error": float(volatility_se),
            "VaR_standard_error": float(-stats.norm.ppf(0.05) * volatility_se * position),
        }

    # SciPy's t scale is NOT its standard deviation.
    nu, location, scale = stats.t.fit(returns)
    results["t_parameters"] = {"nu": float(nu), "mu": float(location), "sigma_scale": float(scale)}
    var["t_MLE"] = float(-stats.t.ppf(0.05, df=nu, loc=location, scale=scale) * position)
    var["historical"] = historical_risk(returns * position)["VaR"]
    results["VaR_dollars"] = var
    results["ranking_smallest_first"] = sorted(var, key=var.get)
    results["EW_VaR_gap"] = var["normal_EW_0.94"] - var["normal_EW_0.97"]
    # A two-normal mixture can be leptokurtic even if each regime is normal.
    share = recent_days / len(returns)
    old_variance = returns[:-recent_days].var(ddof=1)
    new_variance = returns[-recent_days:].var(ddof=1)
    results["zero_mean_normal_mixture_excess_kurtosis"] = float(
        3 * ((1 - share) * old_variance ** 2 + share * new_variance ** 2)
        / ((1 - share) * old_variance + share * new_variance) ** 2 - 3)
    save_results(2, results)


if __name__ == "__main__":
    main()
