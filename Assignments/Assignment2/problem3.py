"""Problem 3: historical and normal risk of concentrated/diversified bonds."""

import pandas as pd

from helpers import FOLDER, save_results
from risk_library import historical_risk, moments, normal_var


def main():
    returns = pd.read_csv(FOLDER / "problem3.csv")
    loss_a = returns.A < -0.2
    loss_b = returns.B < -0.2
    positions = {"1m_A": 1_000_000 * returns.A,
                 "1m_B": 1_000_000 * returns.B,
                 "2m_A": 2_000_000 * returns.A,
                 "1m_A_plus_1m_B": 1_000_000 * (returns.A + returns.B)}
    results = {"moments": {asset: moments(returns[asset]) for asset in ["A", "B"]},
               "large_loss_counts": {"A": int(loss_a.sum()), "B": int(loss_b.sum()),
                                     "either": int((loss_a | loss_b).sum()),
                                     "both": int((loss_a & loss_b).sum())},
               "positions": {}}
    for name, pnl in positions.items():
        results["positions"][name] = {
            "historical_5_percent": historical_risk(pnl, 0.05),
            "historical_1_percent": historical_risk(pnl, 0.01),
            "normal_VaR_5_percent": normal_var(pnl, 0.05),
        }
    save_results(3, results)


if __name__ == "__main__":
    main()
