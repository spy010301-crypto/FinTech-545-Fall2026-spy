"""Portfolio optimization and realized return attribution for tests 10 and 11."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root


def risk_parity(covariance, budget):
    """Find positive holdings with equal variance contribution per unit of budget."""
    budget = np.asarray(budget) / np.sum(budget)

    def equations(log_holdings):
        holdings = np.exp(log_holdings)
        return holdings * (covariance @ holdings) - budget

    start = np.log(np.sqrt(budget / np.diag(covariance)))
    solution = root(equations, start, tol=1e-11)
    if not solution.success or np.max(np.abs(equations(solution.x))) > 1e-10:
        raise RuntimeError(solution.message)
    holdings = np.exp(solution.x)
    return holdings / holdings.sum()


def maximum_sharpe(covariance, means, risk_free, lower, upper):
    excess = means - risk_free

    def objective(weights):
        volatility = np.sqrt(weights @ covariance @ weights)
        extra_return = weights @ excess
        gradient = -excess / volatility + extra_return * (covariance @ weights) / volatility**3
        return -extra_return / volatility, gradient

    count = len(means)
    result = minimize(
        objective, np.full(count, 1 / count), jac=True, method="SLSQP",
        bounds=[(lower, upper)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones(count)},
        options={"ftol": 1e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def attribution(start_weights, stock_returns, factor_returns, betas):
    """Apply buy-and-hold weights, Carino linking, and volatility attribution."""
    stock_growth = 1 + stock_returns.to_numpy()
    position_values = np.vstack([start_weights, stock_growth.cumprod(axis=0) * start_weights])
    before_day = position_values[:-1]
    weights = before_day / before_day.sum(axis=1, keepdims=True)
    portfolio = np.sum(weights * stock_returns.to_numpy(), axis=1)
    factor_contributions = (weights @ betas) * factor_returns.to_numpy()
    alpha = portfolio - factor_contributions.sum(axis=1)
    daily_contributions = np.column_stack([factor_contributions, alpha])

    def compound(returns):
        return np.expm1(np.log1p(returns).sum(axis=0))

    def carino(returns):
        # log(1+r)/r has limit 1 when r is zero.
        returns = np.asarray(returns)
        return np.divide(np.log1p(returns), returns, out=np.ones_like(returns), where=returns != 0)

    total_return = compound(portfolio)
    link = carino(portfolio) / carino(total_return)
    linked_returns = link @ daily_contributions
    centered_portfolio = portfolio - portfolio.mean()
    slopes = centered_portfolio @ daily_contributions / (centered_portfolio @ centered_portfolio)
    portfolio_sd = portfolio.std(ddof=1)

    result = pd.DataFrame([
        np.r_[compound(factor_returns.to_numpy()), compound(alpha), total_return],
        np.r_[linked_returns, total_return],
        np.r_[slopes * portfolio_sd, portfolio_sd],
    ], columns=[*factor_returns.columns, "Alpha", "Portfolio"])
    result.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    return result
