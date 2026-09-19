"""Run Tests 1.1–7.6 and compare saved outputs with the supplied answers."""

from pathlib import Path

import numpy as np
import pandas as pd

from risk_functions import (
    calculate_returns, chol_psd, covariance_to_correlation, ew_covariance,
    fit_nig_mle, fit_nig_moments, fit_normal, fit_t, fit_t_regression,
    higham_psd, missing_covariance, near_psd, pca_root, simulate_normal,
    simulate_pca, t_aicc,
)


FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = FOLDER.parent / "data"
OUTPUT_FOLDER = FOLDER / "outputs"
SAMPLE_COUNT = 100_000


def read_data(filename):
    return pd.read_csv(DATA_FOLDER / filename, float_precision="round_trip")


def compare_output(test, filename, result, theoretical_covariance=None):
    if isinstance(result, dict):
        result = pd.DataFrame([result])
    elif not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result, columns=[f"x{index + 1}" for index in range(result.shape[1])])
    result.to_csv(OUTPUT_FOLDER / filename, index=False)

    # Read back the saved file: these are the values the user will actually see.
    actual = pd.read_csv(OUTPUT_FOLDER / filename, float_precision="round_trip")
    expected = read_data(filename)
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
        raise ValueError(f"Test {test}: output shape or columns do not match.")

    labels_match = True
    if "Date" in expected:
        labels_match = actual["Date"].equals(expected["Date"])
        actual = actual.drop(columns="Date")
        expected = expected.drop(columns="Date")
    actual = actual.to_numpy(dtype=float)
    expected = expected.to_numpy(dtype=float)
    difference = np.abs(actual - expected)

    if theoretical_covariance is None:
        # Local comparison tolerance, not a claimed official grading threshold.
        tolerance = 1e-8 + 1e-6 * np.abs(expected)
        passed = labels_match and np.all(difference <= tolerance)
        comparison = "numeric tolerance"
        largest_error_ratio = np.max(difference / tolerance)
    else:
        # NumPy and Julia generate different random samples, even with seed 1234.
        # For normal samples, Var(sample_cov[i,j]) = (Cii*Cjj + Cij**2)/(n-1).
        variance = np.diag(theoretical_covariance)
        standard_error = np.sqrt(
            (np.outer(variance, variance) + theoretical_covariance ** 2) / (SAMPLE_COUNT - 1)
        )
        tolerance = 5 * np.sqrt(2) * standard_error + 1e-12
        theory_tolerance = 5 * standard_error + 1e-12
        largest_error_ratio = max(
            np.max(difference / tolerance),
            np.max(np.abs(actual - theoretical_covariance) / theory_tolerance),
            np.max(np.abs(expected - theoretical_covariance) / theory_tolerance),
        )
        passed = largest_error_ratio <= 1
        comparison = "Monte Carlo: 5 standard errors"

    status = "PASS" if passed else "FAIL"
    print(f"{test:>3}  {status:4}  max difference = {difference.max():.3g}  ({comparison})")
    return {"test": test, "status": status, "output": filename, "comparison": comparison,
            "max_absolute_difference": difference.max(), "max_error_over_tolerance": largest_error_ratio}


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    results = []

    # Test 1: missing-data covariance and correlation.
    data = read_data("test1.csv")
    for test, skip_missing, correlation in [
        ("1.1", True, False), ("1.2", True, True),
        ("1.3", False, False), ("1.4", False, True),
    ]:
        result = missing_covariance(data, skip_missing, correlation)
        results.append(compare_output(test, f"testout_{test}.csv", result))

    # Test 2: exponentially weighted covariance and correlation.
    data = read_data("test2.csv")
    covariance_97 = ew_covariance(data, 0.97)
    covariance_94 = ew_covariance(data, 0.94)
    correlation_94 = covariance_to_correlation(covariance_94)
    standard_deviation_97 = np.sqrt(np.diag(covariance_97))
    combined_covariance = correlation_94 * np.outer(standard_deviation_97, standard_deviation_97)
    for test, result in [("2.1", covariance_97), ("2.2", correlation_94), ("2.3", combined_covariance)]:
        results.append(compare_output(test, f"testout_{test}.csv", result))

    # Test 3: the previous expected-output files are the specified inputs here.
    covariance = read_data("testout_1.3.csv").to_numpy()
    correlation = read_data("testout_1.4.csv").to_numpy()
    for test, result in [
        ("3.1", near_psd(covariance)), ("3.2", near_psd(correlation)),
        ("3.3", higham_psd(covariance)), ("3.4", higham_psd(correlation)),
    ]:
        results.append(compare_output(test, f"testout_{test}.csv", result))

    # Test 4: positive-semidefinite Cholesky factorization.
    covariance = read_data("testout_3.1.csv").to_numpy()
    results.append(compare_output("4.1", "testout_4.1.csv", chol_psd(covariance)))

    # Test 5: each output is a covariance from 100,000 simulated observations.
    covariance_pd = read_data("test5_1.csv").to_numpy()
    covariance_psd = read_data("test5_2.csv").to_numpy()
    covariance_non_psd = read_data("test5_3.csv").to_numpy()
    for test, covariance, repair, target in [
        ("5.1", covariance_pd, near_psd, covariance_pd),
        ("5.2", covariance_psd, near_psd, covariance_psd),
        ("5.3", covariance_non_psd, near_psd, near_psd(covariance_non_psd)),
        ("5.4", covariance_non_psd, higham_psd, higham_psd(covariance_non_psd)),
    ]:
        samples = simulate_normal(covariance, SAMPLE_COUNT, repair=repair)
        result = np.cov(samples, rowvar=False, ddof=1)
        results.append(compare_output(test, f"testout_{test}.csv", result, target))

    samples = simulate_pca(covariance_psd, SAMPLE_COUNT, explained_variance=0.99)
    root = pca_root(covariance_psd, explained_variance=0.99)
    result = np.cov(samples, rowvar=False, ddof=1)
    results.append(compare_output("5.5", "testout_5.5.csv", result, root @ root.T))

    # Test 6: preserve Date and the original asset-column order.
    prices = read_data("test6.csv")
    results.append(compare_output("6.1", "testout6_1.csv", calculate_returns(prices)))
    results.append(compare_output("6.2", "testout6_2.csv", calculate_returns(prices, log_returns=True)))

    # Test 7: normal, Student-t, regression, AICc, and the two NIG fits.
    normal_data = read_data("test7_1.csv")["x1"].to_numpy()
    t_data = read_data("test7_2.csv")["x1"].to_numpy()
    regression_data = read_data("test7_3.csv")
    nig_data = read_data("test7_5.csv")["x1"].to_numpy()
    t_parameters = fit_t(t_data)
    results.append(compare_output("7.1", "testout7_1.csv", fit_normal(normal_data)))
    results.append(compare_output("7.2", "testout7_2.csv", t_parameters))
    regression_parameters = fit_t_regression(regression_data.drop(columns="y"), regression_data["y"])
    results.append(compare_output("7.3", "testout7_3.csv", regression_parameters))
    results.append(compare_output("7.4", "testout7_4.csv", {"AICC": t_aicc(t_data, t_parameters)}))
    results.append(compare_output("7.5", "testout7_5.csv", fit_nig_moments(nig_data)))
    results.append(compare_output("7.6", "testout7_6.csv", fit_nig_mle(nig_data)))

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_FOLDER / "comparison.csv", index=False)
    failures = int((summary["status"] == "FAIL").sum())
    print(f"\n{len(summary) - failures}/{len(summary)} passed the local comparison criteria.")
    print("Simulation tests use statistical agreement, not an exact CSV match.")
    print(f"Outputs: {OUTPUT_FOLDER}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
