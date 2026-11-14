"""Problem 4: a Sharpe optimizer can concentrate in short-put crash risk."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import ROOT, FIGURES, RESULTS, save_results
from risk_library import moments, historical_risk, normal_risk
from portfolio_library import (
    maximum_sharpe, efficient_frontier, fractional_expected_shortfall, ols,
)


def portfolio_summary(returns, weights, risk_free):
    """Monthly risk per $1 invested, and annualized arithmetic Sharpe ratio."""
    portfolio_returns = returns @ weights
    mean = portfolio_returns.mean()
    volatility = portfolio_returns.std(ddof=1)
    result = {"weights": weights, "mean_monthly": mean, "volatility_monthly": volatility,
              "sharpe_annual": np.sqrt(12) * (mean - risk_free) / volatility}
    for alpha in [0.05, 0.01]:
        result[str(alpha)] = {
            "historical": historical_risk(portfolio_returns, alpha),
            "normal": normal_risk(mean, volatility, alpha),
            "fractional_tail_ES_sensitivity": fractional_expected_shortfall(portfolio_returns, alpha),
        }
    return result


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "problem4.csv")
    names = ["EQ", "BOND", "STK", "PUTW"]
    returns = data[names].to_numpy()
    mean = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False)
    risk_free = 0.03 / 12  # Arithmetic annualization throughout: mean*12, SD*sqrt(12).
    assets = {}
    for column, name in enumerate(names):
        values = returns[:, column]
        worst = np.argmin(values)
        remaining = np.delete(values, worst)
        assets[name] = {
            **moments(values), "standard_deviation": values.std(ddof=1),
            "sharpe_annual": np.sqrt(12) * (values.mean() - risk_free) / values.std(ddof=1),
            "mean_standard_error": values.std(ddof=1) / np.sqrt(len(values)),
            "worst_month": data.Month.iloc[worst], "worst_return": values[worst],
            "worst_centered_variance_share": (values[worst] - values.mean())**2 / np.sum((values - values.mean())**2),
            "worst_centered_fourth_moment_share": (values[worst] - values.mean())**4 / np.sum((values - values.mean())**4),
            "without_worst_moments": moments(remaining),
            "without_worst_mean": remaining.mean(),
            "without_worst_sharpe": np.sqrt(12) * (remaining.mean() - risk_free) / remaining.std(ddof=1),
        }

    # Recorded after the descriptive statistics but before portfolio optimization.
    prediction = (
        "PUTW's high Sharpe suggests a majority (>50%) long-only weight. Its very negative "
        "skew and one large crash suggest historical 1% ES will exceed same-moment normal ES."
    )
    settings = {"unconstrained": {"long_only": False}, "long_only": {},
                "putw_capped_20pct": {"cap": 0.20}}
    portfolios = {}
    for label, options in settings.items():
        weights = maximum_sharpe(mean, covariance, risk_free, **options)
        portfolios[label] = portfolio_summary(returns, weights, risk_free)

    excess_eq = returns[:, 0] - risk_free
    excess_putw = returns[:, 3] - risk_free
    capm = ols(excess_putw, excess_eq, ["EQ"])
    down_months = excess_eq < -0.05
    downside_capm = ols(excess_putw[down_months], excess_eq[down_months], ["EQ"])

    # Resample all four assets together so their cross-asset dependence is kept.
    # Six-month circular blocks also retain short stretches of serial dependence.
    rng = np.random.default_rng(545)
    repetitions, block_length = 500, 6
    samples = {label: [] for label in settings}
    invalid_tangencies = {label: 0 for label in settings}
    for _ in range(repetitions):
        starts = rng.integers(0, len(returns), size=len(returns) // block_length)
        indices = ((starts[:, None] + np.arange(block_length)) % len(returns)).ravel()
        sample = returns[indices]
        for label, options in settings.items():
            try:
                weights = maximum_sharpe(sample.mean(axis=0), np.cov(sample, rowvar=False), risk_free, **options)
            except ValueError:
                # Some unconstrained resamples have no finite positive-Sharpe
                # tangent. Do not report the normalized negative-Sharpe minimum.
                if label != "unconstrained":
                    raise
                invalid_tangencies[label] += 1
                continue
            samples[label].append(weights)
    stability = {}
    for label, weights in samples.items():
        weights = np.asarray(weights)
        stability[label] = {
            "valid_draws": len(weights),
            "invalid_no_finite_tangency_count": invalid_tangencies[label],
            "median": np.median(weights, axis=0),
            "percentile_2_5": np.percentile(weights, 2.5, axis=0),
            "percentile_97_5": np.percentile(weights, 97.5, axis=0),
            "share_putw_above_50pct": np.mean(weights[:, 3] > 0.5),
        }
        pd.DataFrame(weights, columns=names).to_csv(RESULTS / f"problem4_bootstrap_{label}.csv", index=False)

    worst_putw = np.argmin(returns[:, 3])
    without_crash = np.delete(returns, worst_putw, axis=0)
    crash_sensitivity = {
        label: maximum_sharpe(without_crash.mean(axis=0), np.cov(without_crash, rowvar=False), risk_free, **options)
        for label, options in settings.items()
    }

    frontier = efficient_frontier(mean, covariance)
    frontier_frame = pd.DataFrame({
        "mean_monthly": [point["mean"] for point in frontier],
        "volatility_monthly": [point["volatility"] for point in frontier],
        **{name: [point["weights"][j] for point in frontier] for j, name in enumerate(names)},
    })
    frontier_frame.to_csv(RESULTS / "problem4_frontier.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(frontier_frame.volatility_monthly * np.sqrt(12) * 100,
             frontier_frame.mean_monthly * 12 * 100, label="Long-only efficient frontier")
    risk = np.linspace(0, max(np.sqrt(np.diag(covariance))) * np.sqrt(12) * 1.04, 150)
    tangent = portfolios["long_only"]
    plt.plot(risk * 100, (0.03 + tangent["sharpe_annual"] * risk) * 100, "--", label="Capital market line")
    for j, name in enumerate(names):
        x, y = np.sqrt(covariance[j, j] * 12) * 100, mean[j] * 12 * 100
        plt.scatter(x, y)
        plt.annotate(name, (x, y), xytext=(6, 5), textcoords="offset points")
    plt.scatter(tangent["volatility_monthly"] * np.sqrt(12) * 100,
                tangent["mean_monthly"] * 12 * 100, marker="*", s=130, color="black", label="Long-only tangency")
    plt.xlabel("Annualized standard deviation (%)")
    plt.ylabel("Annualized arithmetic mean return (%)")
    plt.title("Problem 4: long-only frontier and capital market line")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES / "problem4_frontier.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(excess_eq * 100, excess_putw * 100, s=15, alpha=0.7, label="Monthly observations")
    x = np.linspace(excess_eq.min(), excess_eq.max(), 100)
    plt.plot(x * 100, (capm["coefficients"]["alpha"] + capm["coefficients"]["EQ"] * x) * 100,
             label="Full-sample CAPM")
    plt.xlabel("EQ monthly excess return (%)")
    plt.ylabel("PUTW monthly excess return (%)")
    plt.title("Problem 4: the nonlinear insurance payoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "problem4_capm.png", dpi=160)
    plt.close()

    capm.pop("residuals")
    downside_capm.pop("residuals")
    result = {
        "n": len(data), "dates": [data.Month.iloc[0], data.Month.iloc[-1]],
        "asset_order": names, "risk_free_monthly": risk_free,
        "conventions": "Monthly arithmetic returns; mean*12 and SD*sqrt(12); sample variance. Risk is monthly loss per $1. Course order-statistic historical VaR and ES are primary; fractional-tail ES is a sensitivity.",
        "prediction_before_optimization": prediction,
        "assets": assets, "covariance_monthly": covariance, "portfolios": portfolios,
        "capm": capm, "downside_capm": downside_capm,
        "bootstrap": {"repetitions": repetitions, "seed": 545, "block_length": block_length,
                      "convention": "Percentiles use valid finite tangencies only. Unconstrained draws with a nonpositive inverse-covariance excess-mean normalizer are counted and excluded; bounded optimizations retain every draw.",
                      "weights": stability},
        "weights_without_putw_worst_month": crash_sensitivity,
    }
    save_results(4, result)
    print("Problem 4 weights (EQ, BOND, STK, PUTW):")
    for label, values in portfolios.items():
        print(label, np.round(values["weights"], 6), "annual Sharpe", round(values["sharpe_annual"], 6))
    print("PUTW full/downside beta:", capm["coefficients"]["EQ"], downside_capm["coefficients"]["EQ"])


if __name__ == "__main__":
    main()
