"""Ten-day risk: compare delta-normal with full option repricing on simulated paths."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import ROOT, FIGURES, RESULTS, save_results
from risk_library import (book_value, book_greeks, moments, fit_returns,
                          normal_risk, historical_risk, simulate_stock)


def summarize_pnl(pnl):
    result = {'mean': float(np.mean(pnl))}
    for alpha in [0.05, 0.01]:
        result[f'{alpha:.0%}'] = historical_risk(pnl, alpha)
        risk = result[f'{alpha:.0%}']
        risk['ES_to_VaR'] = risk['ES'] / risk['VaR'] if risk['VaR'] != 0 else None
    return result


def main():
    prices = pd.read_csv(ROOT / 'problem3.csv')
    positions = pd.read_csv(ROOT / 'problem3_portfolios.csv')
    returns = prices.ABC.pct_change().dropna().to_numpy()
    stock, rate, horizon = float(prices.ABC.iloc[-1]), 0.0425, 10
    draws, seed = 1_000_000, 1234
    books = {}
    # These Greeks are the allowed inputs to the pre-simulation predictions.
    for name, holdings in positions.groupby('Portfolio', sort=False):
        greeks = book_greeks(holdings, stock, rate)
        books[name] = dict(current_value=float(book_value(holdings, stock, rate)),
                           greeks=greeks)
        print(name, 'initial Greeks:', greeks)

    fitted = fit_returns(returns)
    selected = min(fitted, key=lambda name: fitted[name]['aicc'])
    print('Return moments:', moments(returns))
    print('Distribution fits:', fitted, 'Selected:', selected)
    for name, holdings in positions.groupby('Portfolio', sort=False):
        book = books[name]
        # Delta normal excludes option decay and curvature, with zero expected P&L.
        pnl_sd = abs(book['greeks']['delta']) * stock * returns.std(ddof=1) * np.sqrt(horizon)
        book['delta_normal'] = {f'{alpha:.0%}': normal_risk(0, pnl_sd, alpha)
                                for alpha in [0.05, 0.01]}
        book['no_move_pnl'] = float(book_value(holdings, stock, rate, horizon) - book['current_value'])

    simulations = {}
    for distribution in ['normal', 't']:
        # Separate reproducible draws for each fitted model; never use expected answers.
        future_stock = simulate_stock(stock, fitted[distribution], distribution,
                                      horizon, draws, seed)
        price_change = future_stock - stock
        simulations[distribution] = dict(mean_terminal_stock=float(future_stock.mean()),
                                         minimum_terminal_stock=float(future_stock.min()))
        for name, holdings in positions.groupby('Portfolio', sort=False):
            book = books[name]
            delta, gamma, theta = (book['greeks'][key] for key in ['delta', 'gamma', 'theta'])
            full_pnl = book_value(holdings, future_stock, rate, horizon) - book['current_value']
            delta_gamma = delta * price_change + 0.5 * gamma * price_change**2
            approximations = {
                'full_revaluation': full_pnl,
                'delta_gamma': delta_gamma,
                'delta_gamma_theta': delta_gamma + theta * horizon / 255,
            }
            book[distribution] = {method: summarize_pnl(pnl) for method, pnl in approximations.items()}
            # Ten disjoint simulation batches quantify Monte Carlo noise, not model uncertainty.
            batches = [summarize_pnl(batch) for batch in np.array_split(full_pnl, 10)]
            book[distribution]['monte_carlo_se'] = {
                f'{alpha:.0%}': {measure: float(np.std([batch[f'{alpha:.0%}'][measure]
                                for batch in batches], ddof=1) / np.sqrt(10))
                                for measure in ['VaR', 'ES']} for alpha in [0.05, 0.01]
            }
            print(distribution, name, book[distribution]['full_revaluation'])

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    terminal_prices = np.linspace(stock * 0.60, stock * 1.40, 501)
    curves = pd.DataFrame({'terminal_stock': terminal_prices})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for axis, (name, holdings) in zip(axes.flat, positions.groupby('Portfolio', sort=False)):
        full_pnl = book_value(holdings, terminal_prices, rate, horizon) - books[name]['current_value']
        delta_pnl = books[name]['greeks']['delta'] * (terminal_prices - stock)
        curves[name + '_full'] = full_pnl
        curves[name + '_delta'] = delta_pnl
        axis.plot(terminal_prices, full_pnl, color='#2367a0', label='Full revaluation, 10 days later')
        axis.plot(terminal_prices, delta_pnl, '--', color='#c65328', label='Delta only')
        axis.axhline(0, color='gray', linewidth=0.5)
        axis.axvline(stock, color='gray', linewidth=0.5, linestyle=':')
        axis.set(title=name, xlabel='ABC price in 10 trading days ($)', ylabel='Book P&L ($)')
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'problem3_pnl.png', dpi=160)
    plt.close(fig)
    curves.to_csv(RESULTS / 'problem3_pnl_curves.csv', index=False)
    result = dict(stock=stock, returns_count=len(returns), rate=rate, horizon_days=horizon,
                  trading_days_per_year=255, simulations=draws, seed=seed,
                  return_moments=moments(returns), sample_daily_sd=float(returns.std(ddof=1)),
                  fitted_distributions=fitted, selected_distribution=selected,
                  simulation_diagnostics=simulations, books=books)
    save_results(3, result)


if __name__ == '__main__':
    main()
