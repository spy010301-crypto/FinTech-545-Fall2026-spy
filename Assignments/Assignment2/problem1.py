"""Problem 1: missing histories, correlation matrices, and PSD repairs."""

import numpy as np
import pandas as pd

from helpers import FOLDER, higham, near_psd, save_results


def main():
    returns = pd.read_csv(FOLDER / "problem1.csv").drop(columns="Day")
    observed = returns.notna().astype(int)
    counts = observed.T @ observed
    complete = returns.dropna().corr().to_numpy()
    pairwise = returns.corr().to_numpy()
    standard_deviations = returns.std().to_numpy()  # Each asset's full available history.
    tracking_weights = np.array([-0.4, -0.3, -0.2, -0.1, 1.0])

    def tracking_variance(correlation):
        covariance = correlation * np.outer(standard_deviations, standard_deviations)
        return float(tracking_weights @ covariance @ tracking_weights)

    results = {"asset_order": list(returns.columns),
               "joint_observations": counts.to_numpy().tolist(),
               "complete_days": len(returns.dropna()),
               "full_history_standard_deviations": standard_deviations.tolist()}
    for name, correlation in [("complete", complete), ("pairwise", pairwise)]:
        try:
            np.linalg.cholesky(correlation)
            cholesky = "succeeds"
        except np.linalg.LinAlgError:
            cholesky = "fails: matrix is not positive definite"
        results[name] = {"correlation": correlation.tolist(),
                         "eigenvalues": np.linalg.eigvalsh(correlation).tolist(),
                         "cholesky": cholesky}
    results["pairwise_tracking_variance"] = tracking_variance(pairwise)

    for name, repaired in [("Rebonato_Jackel", near_psd(pairwise)),
                           ("Higham", higham(pairwise))]:
        results[name] = {"correlation": repaired.tolist(),
                         "minimum_eigenvalue": float(np.linalg.eigvalsh(repaired).min()),
                         "frobenius_distance": float(np.linalg.norm(repaired - pairwise, "fro")),
                         "tracking_variance": tracking_variance(repaired)}

    changes = []
    higham_matrix = np.array(results["Higham"]["correlation"])
    for first in range(5):
        for second in range(first + 1, 5):
            changes.append({"pair": f"{returns.columns[first]}/{returns.columns[second]}",
                            "change": float(higham_matrix[first, second] - pairwise[first, second]),
                            "joint_days": int(counts.iloc[first, second])})
    results["Higham_changes_largest_first"] = sorted(changes, key=lambda item: abs(item["change"]), reverse=True)
    results["complete_pairwise_distance"] = float(np.linalg.norm(complete - pairwise, "fro"))
    results["distance_between_repairs"] = float(np.linalg.norm(near_psd(pairwise) - higham_matrix, "fro"))
    save_results(1, results)


if __name__ == "__main__":
    main()
