"""Problem 2: American options with two discrete cash dividends."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from american_options import binomial_price, tree_delta_gamma
from risk_library import gbsm


STOCK = 48.50
STRIKE = 50.0
MATURITY = 0.5
RATE = 0.045
VOLATILITY = 0.30
DIVIDEND_TIMES = [0.10, 0.40]
DIVIDEND_AMOUNTS = [0.25, 1.50]


def price(option_type, american, steps, stock=STOCK, dividends=DIVIDEND_AMOUNTS):
    """Keep the problem's parameters in one place for bumps and convergence."""
    dates = DIVIDEND_TIMES if len(dividends) else []
    return binomial_price(option_type, stock, STRIKE, MATURITY, RATE,
                          VOLATILITY, steps, dates, dividends, american)


def exercise_boundary(steps):
    """Pre-second-dividend call boundary, from exercise = continuation."""
    remaining_time = MATURITY - DIVIDEND_TIMES[1]
    remaining_steps = int(round(steps * remaining_time / MATURITY))

    def exercise_minus_hold(stock):
        hold = binomial_price("call", stock - DIVIDEND_AMOUNTS[1], STRIKE,
                              remaining_time, RATE, VOLATILITY,
                              remaining_steps, american=True)
        return stock - STRIKE - hold

    return float(brentq(exercise_minus_hold, STRIKE, 2 * STRIKE))


def main():
    # Predictions were recorded before pricing; they are discussed in the PDF.
    results = {"parameters": {"stock": STOCK, "strike": STRIKE,
               "maturity": MATURITY, "rate": RATE, "volatility": VOLATILITY,
               "dividend_times": DIVIDEND_TIMES, "dividend_amounts": DIVIDEND_AMOUNTS}}
    results["dividend_hurdles"] = [float(STRIKE * (1 - np.exp(-RATE * (MATURITY - time))))
                                  for time in DIVIDEND_TIMES]

    convergence = []
    for steps in [25, 50, 75, 100, 105, 150, 200, 250, 300]:
        convergence.append({"steps": steps,
                            "american_call": price("call", True, steps),
                            "american_put": price("put", True, steps)})
    results["convergence"] = convergence

    # Use a finer tree for final prices and dividend sensitivities. Keep the
    # exercise schedule unchanged for the N=100 stock-bump demonstration.
    steps = 300
    results["pricing_steps"] = steps
    present_value_dividends = float(np.sum(np.array(DIVIDEND_AMOUNTS)
                                          * np.exp(-RATE * np.array(DIVIDEND_TIMES))))
    results["present_value_dividends"] = present_value_dividends
    option_prices = {}
    premiums = {}
    for option_type in ["call", "put"]:
        european_no_dividend = float(gbsm(option_type, STOCK, STRIKE, MATURITY, RATE, VOLATILITY))
        european_no_dividend_tree = price(option_type, False, steps, dividends=[])
        american_no_dividend = price(option_type, True, steps, dividends=[])
        european_dividend = price(option_type, False, steps)
        american_dividend = price(option_type, True, steps)
        adjusted_gbsm = float(gbsm(option_type, STOCK - present_value_dividends,
                                   STRIKE, MATURITY, RATE, VOLATILITY))
        option_prices[option_type] = {
            "european_no_dividend_gbsm": european_no_dividend,
            "european_no_dividend_tree": european_no_dividend_tree,
            "american_no_dividend_tree": american_no_dividend,
            "european_dividend_tree": european_dividend,
            "american_dividend_tree": american_dividend,
            "european_pv_adjusted_gbsm": adjusted_gbsm,
            "pv_adjusted_gbsm_minus_dividend_tree": adjusted_gbsm - european_dividend}
        # Matched trees isolate the exercise right from GBSM/tree discretization.
        premiums[option_type] = {
            "without_dividends_matched_tree": american_no_dividend - european_no_dividend_tree,
            "without_dividends_tree_minus_gbsm": american_no_dividend - european_no_dividend,
            "with_dividends_matched_tree": american_dividend - european_dividend}
    results["prices"] = option_prices
    results["early_exercise_premiums"] = premiums

    results["greek_steps"] = 100
    results["stock_bump_greeks"] = {}
    for option_type in ["call", "put"]:
        results["stock_bump_greeks"][option_type] = {
            str(bump): tree_delta_gamma(STOCK, bump,
                                       lambda stock: price(option_type, True, 100, stock))
            for bump in [1, 0.1, 0.01, 0.001]}

    dividend_bump = 0.01
    results["dividend_bump"] = dividend_bump
    results["dividend_sensitivities"] = {}
    for option_type in ["call", "put"]:
        for american in [False, True]:
            name = ("american_" if american else "european_") + option_type
            sensitivities = []
            for dividend_index in range(2):
                lower = DIVIDEND_AMOUNTS.copy()
                upper = DIVIDEND_AMOUNTS.copy()
                lower[dividend_index] -= dividend_bump
                upper[dividend_index] += dividend_bump
                sensitivity = (price(option_type, american, steps, dividends=upper)
                               - price(option_type, american, steps, dividends=lower)) / (2 * dividend_bump)
                sensitivities.append(sensitivity)
            results["dividend_sensitivities"][name] = sensitivities

    results["call_exercise_boundary"] = {
        "tree_N100": exercise_boundary(100), "tree_N300": exercise_boundary(300),
        "continuous_gbsm_after_dividend": float(brentq(
            lambda stock: stock - STRIKE - gbsm("call", stock - DIVIDEND_AMOUNTS[1],
                STRIKE, MATURITY - DIVIDEND_TIMES[1], RATE, VOLATILITY), STRIKE, 2 * STRIKE))}

    results["conventions"] = {
        "tree": "CRR risk-neutral diffusion between exact dividend dates; independent subtrees after each dividend; no interpolation",
        "stock_floor": "Dividend is capped at stock value, so post-dividend stock is nonnegative; the cap does not bind on these grids",
        "premiums": "Matched European and American trees isolate early exercise; the separate GBSM comparison includes discretization error",
        "sensitivities": "Central dividend bump of $0.01 with N=300; stock bumps at N=100; rates and volatility held fixed"}

    output = Path(__file__).parent / "results" / "problem2_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
