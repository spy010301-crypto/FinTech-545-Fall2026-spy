"""Compute tests 8.1–13.9, save the results, and compare to the course CSVs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from option_models import american_greeks, american_price, european_greeks
from portfolio_models import attribution, maximum_sharpe, risk_parity
from risk_models import (
    copula_log_likelihood, distribution_risk, fit_multivariate_student,
    fit_normal, fit_student, kendall_correlation, profile_degrees, sample_risk,
    simulate_portfolio, simulation_error,
)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def read_data(name):
    return pd.read_csv(DATA / f"{name}.csv")


def first_column(name):
    return read_data(name).iloc[:, 0].to_numpy()


def calculate_risk_tests():
    normal = fit_normal(first_column("test7_1"))
    student = fit_student(first_column("test7_2"))
    # Draw 10,000 scenarios per simulation as in the course test.
    random = np.random.default_rng(8)
    results = {}
    errors = {}
    for first, measure in [(1, "VaR"), (4, "ES")]:
        columns = [f"{measure} Absolute", f"{measure} Diff from Mean"]
        for offset, model in enumerate([normal, student]):
            centered = model.dist(*model.args, scale=model.kwds["scale"])
            values = [distribution_risk(model, measure), distribution_risk(centered, measure)]
            results[f"8.{first + offset}"] = pd.DataFrame([values], columns=columns)
        draws = student.ppf(random.random(10_000))
        test = f"8.{first + 2}"
        values = [sample_risk(draws, measure), sample_risk(draws - draws.mean(), measure)]
        results[test] = pd.DataFrame([values], columns=columns)
        errors[test] = np.array([[simulation_error(draws, measure),
                                  simulation_error(draws, measure, centered=True)]])
    return results, errors


def portfolio_risk_table(pnl, names, exposures):
    """Summarize risk by asset and for the same joint scenarios summed together."""
    rows, errors = [], []
    for name, scenarios, exposure in zip([*names, "Total"],
                                         [*pnl.T, pnl.sum(axis=1)],
                                         [*exposures, exposures.sum()]):
        var, es = [sample_risk(scenarios, measure) for measure in ("VaR", "ES")]
        var_error, es_error = [simulation_error(scenarios, measure) for measure in ("VaR", "ES")]
        rows.append([name, var, es, var / exposure, es / exposure])
        errors.append([var_error, es_error, var_error / exposure, es_error / exposure])
    columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
    return pd.DataFrame(rows, columns=columns), np.asarray(errors)


def gaussian_portfolio_test():
    positions = read_data("test9_1_portfolio")
    names = positions["Stock"]
    returns = read_data("test9_1_returns")[names]
    fitters = {"normal": fit_normal, "t": fit_student}
    margins = [fitters[kind.lower()](returns[name].to_numpy())
               for name, kind in zip(names, positions["Distribution"])]
    # Test 9 uses Spearman correlation directly as the latent correlation.
    correlation = returns.corr(method="spearman").to_numpy()
    exposures = (positions["Holding"] * positions["Starting Price"]).to_numpy()
    pnl = simulate_portfolio(margins, correlation, exposures, 100_000, seed=1234)
    return portfolio_risk_table(pnl, names, exposures)


def optimization_tests():
    parity_covariance = read_data("test5_2").to_numpy()
    means = first_column("test10_3_means")
    # The spreadsheet says test5_2 here, but the course generator uses test5_3.
    sharpe_covariance = read_data("test5_3").to_numpy()
    weights = {
        "10.1": risk_parity(parity_covariance, [1, 1, 1, 1, 1]),
        "10.2": risk_parity(parity_covariance, [1, 1, 1, 1, 0.5]),
        "10.3": maximum_sharpe(sharpe_covariance, means, 0.04, 0, 1),
        "10.4": maximum_sharpe(sharpe_covariance, means, 0.04, 0.1, 0.5),
    }
    return {test: pd.DataFrame({"W": values}) for test, values in weights.items()}


def attribution_tests():
    stock_returns = read_data("test11_1_returns")
    asset_result = attribution(first_column("test11_1_weights"), stock_returns,
                               stock_returns, np.eye(stock_returns.shape[1]))
    stock_returns = read_data("test11_2_stock_returns")
    factor_returns = read_data("test11_2_factor_returns")
    betas = read_data("test11_2_beta").set_index("Stock").loc[stock_returns.columns, factor_returns.columns]
    factor_result = attribution(first_column("test11_2_weights"), stock_returns,
                                factor_returns, betas.to_numpy())
    return {"11.1": asset_result.drop(columns="Alpha"), "11.2": factor_result}


def option_tests():
    results = {"12.1": [], "12.2": [], "12.3": []}
    for row in read_data("test12_1").dropna(subset=["ID"]).to_dict("records"):
        is_call = row["Option Type"] == "Call"
        rate = row["RiskFreeRate"]
        parameters = (
            row["Underlying"], row["Strike"], row["DaysToMaturity"] / row["DayPerYear"],
            rate, rate - row["DividendRate"], row["ImpliedVol"],
        )
        results["12.1"].append({"ID": int(row["ID"]), **european_greeks(is_call, *parameters)})
        results["12.2"].append({"ID": int(row["ID"]), **american_greeks(is_call, parameters)})

    for row in read_data("test12_3").dropna(subset=["ID"]).to_dict("records"):
        rate = row["RiskFreeRate"]
        value = american_price(
            row["Option Type"] == "Call", row["Underlying"], row["Strike"],
            row["DaysToMaturity"] / row["DayPerYear"], rate, rate, row["ImpliedVol"],
            steps=2 * int(row["DaysToMaturity"]),
            dividend_steps=tuple(2 * int(day) for day in row["DividendDates"].split(",")),
            dividends=tuple(float(amount) for amount in row["DividendAmts"].split(",")),
        )
        results["12.3"].append({"ID": int(row["ID"]), "Value": value})
    return {test: pd.DataFrame(rows) for test, rows in results.items()}


def student_copula_tests():
    positions = read_data("test13_portfolio")
    names = positions["Stock"]
    returns = read_data("test13_returns")[names].to_numpy()
    correlation = kendall_correlation(returns)
    location, scale, joint_degrees, joint_ll = fit_multivariate_student(returns, correlation)
    margins = [fit_student(column) for column in returns.T]
    uniforms = np.column_stack([margin.cdf(column) for margin, column in zip(margins, returns.T)])
    gaussian_ll = copula_log_likelihood(uniforms, correlation)
    degrees, student_ll = profile_degrees(
        lambda candidate: copula_log_likelihood(uniforms, correlation, candidate)
    )
    parameters = np.array([0, 1])  # The two copulas share the same pre-estimated correlation.
    likelihoods = np.array([gaussian_ll, student_ll])
    count = len(returns)
    aicc = -2 * likelihoods + 2 * parameters + 2 * parameters * (parameters + 1) / (count - parameters - 1)
    bic = -2 * likelihoods + parameters * np.log(count)
    first, second = np.triu_indices(len(names), k=1)
    rho = correlation[first, second]
    tail_dependence = 2 * stats.t.cdf(-np.sqrt((degrees + 1) * (1 - rho) / (1 + rho)), degrees + 1)
    exposures = positions["currentValue"].to_numpy()
    pnl = simulate_portfolio(margins, correlation, exposures, 100_000, 13, degrees)
    risk_table, errors = portfolio_risk_table(pnl, names, exposures)
    matrix_columns = [f"x{index}" for index in range(1, len(names) + 1)]
    return {
        "13.1": pd.DataFrame(correlation, columns=matrix_columns),
        "13.2": pd.DataFrame({"mu": location}),
        "13.3": pd.DataFrame(scale, columns=matrix_columns),
        "13.4": pd.DataFrame({"nu": [joint_degrees], "ll": [joint_ll]}),
        "13.5": pd.DataFrame({"ll": [gaussian_ll]}),
        "13.6": pd.DataFrame({"nu": [degrees], "ll": [student_ll]}),
        "13.7": pd.DataFrame({"Copula": ["Gaussian", "T"], "LL": likelihoods,
                              "K": parameters, "AICC": aicc, "BIC": bic}),
        "13.8": pd.DataFrame({"I": first + 1, "J": second + 1, "Rho": rho, "Lambda": tail_dependence}),
        "13.9": risk_table,
    }, errors


def compare(test, actual, expected, monte_carlo_error=None):
    """Check the CSV schema and each cell. Random cases have their own error test."""
    method = "numeric tolerance" if monte_carlo_error is None else "Monte Carlo: 5 standard errors"
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
        return "FAIL", "CSV column or row mismatch", np.nan, np.nan

    identifiers = {"Value"} if test.startswith("11.") else {"ID", "Stock", "Copula", "K", "I", "J"}
    exact = [column for column in expected if column in identifiers or not pd.api.types.is_numeric_dtype(expected[column])]
    for column in exact:
        if not np.array_equal(actual[column], expected[column]):
            return "FAIL", f"Identifier mismatch in {column}", np.nan, np.nan

    numeric = [column for column in expected if column not in exact]
    target = expected[numeric].to_numpy(float)
    values = actual[numeric].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        return "FAIL", "Non-finite numeric value", np.nan, np.nan

    difference = np.abs(values - target)
    tolerance = 1e-8 + 1e-6 * np.abs(target)
    if monte_carlo_error is not None:
        # Both CSVs estimate the same risk with independent scenario samples.
        tolerance = 5 * np.sqrt(2) * monte_carlo_error
    ratio = np.max(difference / tolerance)
    return ("PASS" if ratio <= 1 else "FAIL"), method, difference.max(), ratio


def calculate_all():
    """Compute all outputs from test inputs before loading any reference output."""
    results, errors = calculate_risk_tests()
    results["9.1"], errors["9.1"] = gaussian_portfolio_test()
    results.update(optimization_tests())
    results.update(attribution_tests())
    results.update(option_tests())
    student_results, errors["13.9"] = student_copula_tests()
    results.update(student_results)
    return results, errors


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    output_dir = parser.parse_args().output_dir.resolve()
    if output_dir.is_relative_to(HERE.parent) and not output_dir.is_relative_to(HERE):
        parser.error("Choose an output directory inside this part's folder or outside testfiles")

    results, errors = calculate_all()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for test, actual in results.items():
        filename = "testout" + test.replace(".", "_")
        actual.to_csv(output_dir / (filename + ".csv"), index=False)
        status, method, difference, ratio = compare(test, actual, read_data(filename), errors.get(test))
        report.append([test, status, filename + ".csv", method, difference, ratio])
        print(f"{test:>4}  {status}  {method} (error/tolerance {ratio:.5g})")
    columns = ["test", "status", "output", "comparison", "max_absolute_difference", "max_error_over_tolerance"]
    pd.DataFrame(report, columns=columns).to_csv(output_dir / "comparison.csv", index=False)
    passed = sum(row[1] == "PASS" for row in report)
    print(f"{passed}/{len(report)} tests passed.")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
