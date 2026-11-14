"""Course-style binomial trees, including discrete cash dividends.

This follows library/bt_american.jl. Each dividend-date node starts its own
subtree. NumPy evaluates those independent subtrees together for speed;
there is no interpolation or forced recombination across dividend dates.
"""

import numpy as np


def binomial_price(option_type, stock, strike, maturity, rate, volatility,
                   steps, dividend_times=(), dividend_amounts=(), american=True):
    """Price one call or put with annual continuously compounded rates.

    Dividend dates must lie exactly on the grid. Cash is taken off the
    stock immediately after the pre-dividend exercise decision. If a
    dividend exceeds the stock price, the paid dividend is capped at the
    available stock value, leaving a zero stock price (limited liability).
    The cap never binds for the Problem 2 grids used here.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if len(dividend_times) != len(dividend_amounts):
        raise ValueError("Each dividend needs an amount and a date")

    dt = maturity / steps
    dividend_steps = np.rint(np.asarray(dividend_times) / dt).astype(int)
    if not np.allclose(dividend_steps * dt, dividend_times, atol=1e-12, rtol=0):
        raise ValueError("Choose steps that put every dividend exactly on the grid")
    if len(dividend_steps) and (np.any(np.diff(dividend_steps) <= 0)
                               or dividend_steps[0] <= 0
                               or dividend_steps[-1] >= steps):
        raise ValueError("Dividend dates must increase and lie before expiry")

    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    probability_up = (np.exp(rate * dt) - down) / (up - down)
    discount = np.exp(-rate * dt)
    direction = 1 if option_type == "call" else -1

    def payoff(prices):
        return np.maximum(direction * (prices - strike), 0)

    def subtree(start_prices, remaining_steps, remaining_dates, remaining_amounts):
        # One row per independent subtree; columns are its recombining nodes.
        interval = int(remaining_dates[0]) if len(remaining_dates) else remaining_steps
        end_prices = start_prices[:, None] * up ** (2 * np.arange(interval + 1) - interval)

        if len(remaining_dates):
            after_dividend = np.maximum(end_prices - remaining_amounts[0], 0)
            values = subtree(after_dividend.ravel(), remaining_steps - interval,
                             remaining_dates[1:] - interval, remaining_amounts[1:])
            values = values.reshape(end_prices.shape)
            if american:
                values = np.maximum(values, payoff(end_prices))
        else:
            values = payoff(end_prices)

        # Between dividends this is the ordinary CRR backward recursion.
        for time_step in range(interval - 1, -1, -1):
            values = discount * ((1 - probability_up) * values[:, :-1]
                                 + probability_up * values[:, 1:])
            if american:
                prices = start_prices[:, None] * up ** (2 * np.arange(time_step + 1) - time_step)
                values = np.maximum(values, payoff(prices))
        return values[:, 0]

    result = subtree(np.array([float(stock)]), steps, dividend_steps,
                     np.asarray(dividend_amounts, dtype=float))
    return float(result[0])


def tree_delta_gamma(stock, bump, price_function):
    """Central finite differences; a tiny bump does not smooth tree kinks."""
    lower = price_function(stock - bump)
    middle = price_function(stock)
    upper = price_function(stock + bump)
    delta = (upper - lower) / (2 * bump)
    gamma = (upper - 2 * middle + lower) / bump ** 2
    return {"delta": float(delta), "gamma": float(gamma)}
