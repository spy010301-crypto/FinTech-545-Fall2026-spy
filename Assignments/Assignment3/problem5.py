"""Problem 5: distinguish a fund's alpha from its factor exposures."""

import numpy as np
import pandas as pd

from helpers import ROOT, RESULTS, save_results
from portfolio_library import ols


def read_factors(filename):
    """The supplied French files have a header; values are percentages, not decimals."""
    data = pd.read_csv(ROOT / filename)
    data.columns = data.columns.str.strip()
    data["Date"] = pd.to_datetime(data["Date"].astype(str), format="%Y%m%d")
    columns = data.columns.drop("Date")
    data[columns] = data[columns] / 100
    return data


def main():
    RESULTS.mkdir(exist_ok=True)
    funds = pd.read_csv(ROOT / "problem5.csv", parse_dates=["Date"])
    french = read_factors("F-F_Research_Data_Factors_daily.CSV")
    momentum = read_factors("F-F_Momentum_Factor_daily.CSV")
    factors = french.merge(momentum, on="Date", validate="one_to_one").sort_values("Date")
    data = funds.merge(factors, on="Date", validate="one_to_one").sort_values("Date")
    factor_names = ["Mkt-RF", "SMB", "HML", "Mom"]
    days_per_year = 255  # The annualization convention used in this course's examples.

    # The requested most-recent decade is in the factor files, not the fund sample.
    # This is a hypothetical current-premia estimate, not an out-of-sample backtest.
    end = factors.Date.max()
    start = end - pd.DateOffset(years=10)
    recent = factors.loc[(factors.Date > start) & (factors.Date <= end)]
    recent_mean = recent[factor_names + ["RF"]].mean()
    sample_mean = data[factor_names + ["RF"]].mean()

    # Search the entire momentum file, including dates outside both other windows.
    worst = momentum.loc[momentum.Mom.idxmin()]
    results = {}
    coefficient_rows = []
    for fund in ["FUND1", "FUND2"]:
        total = data[fund].to_numpy()
        excess = (data[fund] - data.RF).to_numpy()
        capm = ols(excess, data[["Mkt-RF"]], ["Mkt-RF"])
        residual_correlations = {
            name: np.corrcoef(capm["residuals"], data[name])[0, 1]
            for name in ["SMB", "HML", "Mom"]
        }
        # Prediction was recorded from these CAPM diagnostics before FF3/Carhart.
        prediction = (
            "Momentum should remove much of the apparent FUND1 alpha; "
            "FUND2 alpha should move less because its CAPM momentum residual correlation is small."
        )
        models = {
            "CAPM": capm,
            "FF3": ols(excess, data[factor_names[:3]], factor_names[:3]),
            "Carhart4": ols(excess, data[factor_names], factor_names),
        }
        four = models["Carhart4"]
        beta = pd.Series(four["coefficients"]).reindex(factor_names)
        alpha = four["coefficients"]["alpha"]
        expected_daily = recent_mean.RF + beta @ recent_mean[factor_names]

        # Isolated factor shock: no normal alpha, RF accrual, or residual shock is
        # added; other factors are flat. Dollar loss may be negative for a hedge.
        shock_return = beta.Mom * worst.Mom
        daily_volatility = total.std(ddof=1)
        contributions = {name: beta[name] * sample_mean[name] for name in factor_names}
        alpha_bridge = {
            "market_beta_change_times_market_mean": (beta["Mkt-RF"] - capm["coefficients"]["Mkt-RF"]) * sample_mean["Mkt-RF"],
            **{name: contributions[name] for name in ["SMB", "HML", "Mom"]},
        }
        for model_name, model in models.items():
            for name, coefficient in model["coefficients"].items():
                coefficient_rows.append({"fund": fund, "model": model_name, "coefficient": name,
                                         "estimate": coefficient, "standard_error": model["standard_errors"][name],
                                         "t_statistic": model["t_statistics"][name],
                                         "r_squared": model["r_squared"], "aicc": model["aicc"]})
            model.pop("residuals")
        results[fund] = {
            "daily_mean_excess": excess.mean(), "daily_total_volatility": daily_volatility,
            "daily_excess_volatility": excess.std(ddof=1),
            "annualized_excess_mean": days_per_year * excess.mean(),
            "annualized_total_volatility": np.sqrt(days_per_year) * daily_volatility,
            "sharpe_annual": np.sqrt(days_per_year) * excess.mean() / excess.std(ddof=1),
            "capm_residual_correlations": residual_correlations,
            "prediction_before_multifactor_fit": prediction,
            "models": models,
            "expected_annual_without_alpha": days_per_year * expected_daily,
            "expected_annual_with_alpha": days_per_year * (expected_daily + alpha),
            "realized_annual_arithmetic": days_per_year * total.mean(),
            "realized_annual_geometric": np.expm1(np.log1p(total).sum() * days_per_year / len(total)),
            "momentum_stress": {"return": shock_return, "loss_on_10million": -10000000 * shock_return,
                                "loss_daily_sd_multiple": -shock_return / daily_volatility},
            "sample_factor_return_contributions_daily": contributions,
            "capm_alpha_minus_carhart_alpha": capm["coefficients"]["alpha"] - alpha,
            "alpha_difference_bridge_daily": alpha_bridge,
            "alpha_bridge_sum": sum(alpha_bridge.values()),
        }

    pd.DataFrame(coefficient_rows).to_csv(RESULTS / "problem5_regressions.csv", index=False)
    factor_means = pd.DataFrame({"sample_daily_mean": sample_mean, "recent_10year_daily_mean": recent_mean})
    factor_means.to_csv(RESULTS / "problem5_factor_premia.csv", index_label="factor")
    result = {
        "n_fund_rows": len(funds), "n_aligned_rows": len(data),
        "fund_sample_dates": [str(data.Date.min().date()), str(data.Date.max().date())],
        "recent_factor_window": {"exclusive_start": str(start.date()), "inclusive_end": str(end.date()),
                                 "first_observation": str(recent.Date.min().date()), "n": len(recent)},
        "factor_columns": factor_names, "days_per_year": days_per_year,
        "conventions": "All returns are decimals. OLS uses excess fund returns, an intercept, conventional homoskedastic SEs. AICc k includes error variance. Annual expected/realized comparison uses 255*daily arithmetic mean; CAGR is separate. Recent factor premia postdate the fund sample and are not a backtest.",
        "sample_factor_means_daily": sample_mean.to_dict(),
        "recent_factor_means_daily": recent_mean.to_dict(),
        "worst_momentum_entire_file": {"date": str(worst.Date.date()), "return": worst.Mom},
        "funds": results,
    }
    save_results(5, result)
    print("Problem 5:")
    for fund, values in results.items():
        print(fund, "CAPM -> FF3 -> Carhart daily alpha:",
              [round(values["models"][model]["coefficients"]["alpha"], 8) for model in ["CAPM", "FF3", "Carhart4"]])
        print("Expected annual without/with alpha:", values["expected_annual_without_alpha"], values["expected_annual_with_alpha"])
    print("Worst momentum event:", str(worst.Date.date()), worst.Mom)


if __name__ == "__main__":
    main()
