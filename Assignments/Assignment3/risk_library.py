"""Reusable option and risk functions, translated from the course's Julia examples.

All functions take inputs and return results. Importing this module does not
read assignment data, print answers, fit models, or write any files.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def gbsm(option_type, stock, strike, maturity, rate, volatility, dividend_yield=0.0):
    """European option value. Rates and volatility are annual decimals.

    The stock can be a scalar or an array of simulation prices. The stock model
    uses cost of carry b = rate - dividend_yield in generalized BSM.
    """
    stock = np.asarray(stock, dtype=float)
    sign = 1 if option_type.lower() == 'call' else -1
    if maturity <= 0:
        return np.maximum(sign * (stock - strike), 0.0)
    if volatility == 0:
        forward = stock * np.exp((rate - dividend_yield) * maturity)
        return np.exp(-rate * maturity) * np.maximum(sign * (forward - strike), 0)
    # A zero stock is absorbing; the limiting put value is discounted strike.
    safe_stock = np.maximum(stock, np.finfo(float).tiny)
    d1 = (np.log(safe_stock / strike)
          + (rate - dividend_yield + volatility**2 / 2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return sign * (stock * np.exp(-dividend_yield * maturity) * stats.norm.cdf(sign * d1)
                   - strike * np.exp(-rate * maturity) * stats.norm.cdf(sign * d2))


def option_greeks(option_type, stock, strike, maturity, rate, volatility, dividend_yield=0.0):
    """Delta, gamma, vega per 1.00 volatility, and theta per elapsed year."""
    sign = 1 if option_type.lower() == 'call' else -1
    d1 = (np.log(stock / strike)
          + (rate - dividend_yield + volatility**2 / 2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    discount_stock = np.exp(-dividend_yield * maturity)
    density = stats.norm.pdf(d1)
    return {
        'delta': sign * discount_stock * stats.norm.cdf(sign * d1),
        'gamma': discount_stock * density / (stock * volatility * np.sqrt(maturity)),
        'vega': stock * discount_stock * density * np.sqrt(maturity),
        'theta': (-stock * discount_stock * density * volatility / (2 * np.sqrt(maturity))
                  + sign * dividend_yield * stock * discount_stock * stats.norm.cdf(sign * d1)
                  - sign * rate * strike * np.exp(-rate * maturity) * stats.norm.cdf(sign * d2)),
    }


def implied_volatility(option_type, price, stock, strike, maturity, rate, dividend_yield=0.0):
    """Invert our price with a bracketed root; invalid no-arbitrage prices return None."""
    stock_pv = stock * np.exp(-dividend_yield * maturity)
    strike_pv = strike * np.exp(-rate * maturity)
    if option_type.lower() == 'call':
        lower, upper = max(stock_pv - strike_pv, 0.0), stock_pv
    else:
        lower, upper = max(strike_pv - stock_pv, 0.0), strike_pv
    if price < lower - 1e-10 or price >= upper:
        return None
    if abs(price - lower) <= 1e-10:
        return 0.0
    def difference(volatility):
        return gbsm(option_type, stock, strike, maturity, rate, volatility, dividend_yield) - price
    high = 1.0
    while difference(high) < 0:
        high *= 2
    return float(brentq(difference, 1e-9, high, xtol=1e-12))


def moments(values):
    """Sample variance and standardized central moments (excess kurtosis)."""
    values = np.asarray(values)
    return dict(mean=float(values.mean()), variance=float(values.var(ddof=1)),
                skewness=float(stats.skew(values)),
                excess_kurtosis=float(stats.kurtosis(values, fisher=True)))


def aicc(log_likelihood, parameters, observations):
    """The parameter count includes every estimated parameter, including scale."""
    return float(-2 * log_likelihood + 2 * parameters
                 + 2 * parameters * (parameters + 1) / (observations - parameters - 1))


def fit_returns(returns):
    """Normal and Student t MLE; fit observed means, then set means to zero in simulation."""
    returns = np.asarray(returns)
    mean, normal_sd = stats.norm.fit(returns)
    degrees, location, scale = stats.t.fit(returns)
    normal_loglik = stats.norm.logpdf(returns, mean, normal_sd).sum()
    t_loglik = stats.t.logpdf(returns, degrees, location, scale).sum()
    return {
        'normal': dict(mean=float(mean), sd=float(normal_sd),
                       log_likelihood=float(normal_loglik), aicc=aicc(normal_loglik, 2, len(returns))),
        't': dict(df=float(degrees), location=float(location), scale=float(scale),
                  sd=float(scale * np.sqrt(degrees / (degrees - 2))),
                  log_likelihood=float(t_loglik), aicc=aicc(t_loglik, 3, len(returns))),
    }


def normal_risk(mean, sd, alpha=0.05):
    """Normal VaR/ES, positive for losses; do not floor a negative loss estimate."""
    quantile = stats.norm.ppf(alpha)
    return {'VaR': float(-mean - sd * quantile),
            'ES': float(-mean + sd * stats.norm.pdf(quantile) / alpha)}


def historical_risk(pnl, alpha=0.05):
    """Match library/RiskStats.jl's order statistic and tail-inclusion convention."""
    ordered = np.sort(np.asarray(pnl))
    rank = alpha * len(ordered)
    lower = max(int(np.floor(rank)) - 1, 0)
    upper = max(int(np.ceil(rank)) - 1, 0)
    cutoff = (ordered[lower] + ordered[upper]) / 2
    return {'VaR': float(-cutoff), 'ES': float(-ordered[ordered <= cutoff].mean())}


def book_value(positions, stock, rate, elapsed_days=0, trading_days=255):
    """Value single-share stock/option positions at a scalar or scenario array."""
    value = np.zeros_like(np.asarray(stock, dtype=float))
    for position in positions.itertuples():
        if position.Type == 'Stock':
            value = value + position.Holding * stock
        else:
            maturity = (position.DaysToExpiry - elapsed_days) / trading_days
            value = value + position.Holding * gbsm(
                position.Type, stock, position.Strike, maturity, rate, position.ImpliedVol)
    return value


def book_greeks(positions, stock, rate, trading_days=255):
    """Add position Greeks; the supplied Holding is already the option count."""
    total = dict(delta=0.0, gamma=0.0, theta=0.0)
    for position in positions.itertuples():
        if position.Type == 'Stock':
            total['delta'] += position.Holding
        else:
            greeks = option_greeks(position.Type, stock, position.Strike,
                                    position.DaysToExpiry / trading_days, rate, position.ImpliedVol)
            for name in total:
                total[name] += position.Holding * greeks[name]
    return total


def simulate_stock(stock, fitted, distribution, horizon, draws, seed):
    """Compound independent daily arithmetic returns, with model mean set to zero.

    A fitted t's scale is not its SD. Keep its fitted df/scale; remove only its
    location. Arithmetic models have unbounded downside; reject a run with an
    impossible price instead of silently clipping and changing its mean.
    """
    rng = np.random.default_rng(seed)
    future_stock = np.full(draws, stock, dtype=float)
    for day in range(horizon):
        if distribution == 'normal':
            daily_return = rng.normal(0, fitted['sd'], draws)
        else:
            daily_return = fitted['scale'] * rng.standard_t(fitted['df'], draws)
        if np.any(daily_return <= -1):
            raise ValueError('A simulated arithmetic return is <= -100%; review the return model.')
        future_stock *= 1 + daily_return
    return future_stock
