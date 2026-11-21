"""Course option values and Greeks for test 12."""

import numpy as np
from scipy.stats import norm


def european_greeks(is_call, stock, strike, maturity, rate, carry, volatility):
    """Generalized Black-Scholes-Merton with rho holding dividend yield fixed."""
    sign = 1 if is_call else -1
    sqrt_time = np.sqrt(maturity)
    first = (np.log(stock / strike) + (carry + volatility**2 / 2) * maturity) / (volatility * sqrt_time)
    second = first - volatility * sqrt_time
    stock_discount = np.exp((carry - rate) * maturity)
    strike_discount = np.exp(-rate * maturity)
    stock_term = sign * stock * stock_discount * norm.cdf(sign * first)
    strike_term = sign * strike * strike_discount * norm.cdf(sign * second)
    density = norm.pdf(first)
    return {
        "Value": stock_term - strike_term,
        "Delta": stock_term / stock,
        "Gamma": stock_discount * density / (stock * volatility * sqrt_time),
        "Vega": stock * stock_discount * density * sqrt_time,
        "Rho": maturity * strike_term,
        "Theta": (-stock * stock_discount * density * volatility / (2 * sqrt_time)
                  - (carry - rate) * stock_term - rate * strike_term),
    }


def american_price(is_call, stock, strike, maturity, rate, carry, volatility,
                   steps, dividend_steps=(), dividends=()):
    """CRR backward induction, starting a separate subtree at each cash dividend."""
    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    probability_up = (np.exp(carry * dt) - down) / (up - down)
    discount = np.exp(-rate * dt)
    sign = 1 if is_call else -1

    def payoff(prices):
        return np.maximum(sign * (prices - strike), 0)

    def nodes(start, interval):
        return start[:, None] * up ** (2 * np.arange(interval + 1) - interval)

    def subtree(start, remaining, dates, amounts):
        interval = dates[0] if dates else remaining
        terminal_prices = nodes(start, interval)
        values = payoff(terminal_prices)
        if dates:
            after_dividend = (terminal_prices - amounts[0]).ravel()
            next_dates = tuple(date - interval for date in dates[1:])
            continuation = subtree(after_dividend, remaining - interval, next_dates, amounts[1:])
            values = np.maximum(values, continuation.reshape(values.shape))

        for time in range(interval - 1, -1, -1):
            continuation = discount * ((1 - probability_up) * values[:, :-1]
                                       + probability_up * values[:, 1:])
            values = np.maximum(payoff(nodes(start, time)), continuation)
        return values[:, 0]

    return subtree(np.array([float(stock)]), steps, dividend_steps, dividends)[0]


def american_greeks(is_call, parameters):
    """Use the 500-step tree and the derivative conventions of Test 12.2."""
    def price(values):
        return american_price(is_call, *values, steps=500)

    parameters = np.asarray(parameters)
    price_now = price(parameters)

    def central_difference(index, bump, order=1):
        move = np.zeros_like(parameters)
        move[index] = bump
        high, low = price(parameters + move), price(parameters - move)
        if order == 2:
            return (high - 2 * price_now + low) / bump**2
        return (high - low) / (2 * bump)

    def gradient(index):
        bump = np.cbrt(np.finfo(float).eps) * max(1, abs(parameters[index]))
        return central_difference(index, bump)

    return {
        "Value": price_now, "Delta": gradient(0), "Gamma": central_difference(0, 1.5, order=2),
        "Vega": gradient(5), "Rho": gradient(3), "Theta": gradient(2),
    }
