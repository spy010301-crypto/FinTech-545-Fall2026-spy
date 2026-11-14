"""Dirty option quotes: parity first, then our GBSM implied volatility inversion."""

from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import ROOT, FIGURES, RESULTS, save_results
from risk_library import implied_volatility, option_greeks


def main():
    stock, rate, dividend_yield = 182.40, 0.0425, 0.006
    maturity = (date(2026, 12, 18) - date(2026, 11, 13)).days / 365
    quotes = pd.read_csv(ROOT / 'problem1.csv')
    quotes['Mid'] = (quotes.Bid + quotes.Ask) / 2
    calls = quotes[quotes.Type == 'Call'].set_index('Strike')
    puts = quotes[quotes.Type == 'Put'].set_index('Strike')
    parity = pd.DataFrame(index=calls.index)
    parity['gap'] = (calls.Mid - puts.Mid - stock * np.exp(-dividend_yield * maturity)
                     + parity.index * np.exp(-rate * maturity))
    # The question asks for the FULL sum of spreads, not half-spreads.
    parity['band'] = calls.Ask - calls.Bid + puts.Ask - puts.Bid
    parity['flag'] = parity.gap.abs() > parity.band
    print('Parity BEFORE the IV fits:')
    print(parity.to_string())

    records = []
    for quote in quotes.itertuples():
        result = {'type': quote.Type, 'strike': quote.Strike}
        for label in ['Bid', 'Mid', 'Ask']:
            result[label.lower() + '_iv'] = implied_volatility(
                quote.Type, getattr(quote, label), stock, quote.Strike,
                maturity, rate, dividend_yield)
        greeks = option_greeks(quote.Type, stock, quote.Strike, maturity,
                               rate, result['mid_iv'], dividend_yield)
        result['vega'] = float(greeks['vega'])
        # Vol points = 100 * decimal volatility. This is a local derivative.
        result['cent_iv_points'] = 100 * 0.01 / result['vega']
        records.append(result)
    volatility = pd.DataFrame(records)
    discrepancies = []
    for strike in parity.index[parity.flag]:
        call = volatility[(volatility.type == 'Call') & (volatility.strike == strike)].iloc[0]
        put = volatility[(volatility.type == 'Put') & (volatility.strike == strike)].iloc[0]
        reference_vega = (call.vega + put.vega) / 2
        discrepancies.append(dict(strike=float(strike), gap=float(parity.loc[strike, 'gap']),
                                  reference_vega=float(reference_vega),
                                  approximate_iv_gap_points=float(100 * parity.loc[strike, 'gap'] / reference_vega),
                                  actual_iv_gap_points=float(100 * (call.mid_iv - put.mid_iv))))

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    for kind, color, marker in [('Call', '#2367a0', 'o'), ('Put', '#c65328', 's')]:
        group = volatility[volatility.type == kind]
        axis.plot(group.strike, 100 * group.mid_iv, marker=marker, color=color, label=f'{kind} mid IV')
        axis.plot(group.strike, 100 * group.bid_iv, ':', color=color, alpha=0.65)
        axis.plot(group.strike, 100 * group.ask_iv, ':', color=color, alpha=0.65)
    axis.axvline(175, color='gray', linestyle='--', linewidth=1, label='Parity flag: 175')
    axis.set(xlabel='Strike ($)', ylabel='Implied volatility (%)',
             title='Mid IV with dotted bid/ask IV bounds; missing bids have no IV')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'problem1_implied_volatility.png', dpi=160)
    plt.close(fig)
    parity.to_csv(RESULTS / 'problem1_parity.csv')
    volatility.to_csv(RESULTS / 'problem1_implied_volatility.csv', index=False)
    result = dict(stock=stock, maturity_years=maturity, rate=rate, dividend_yield=dividend_yield,
                  parity=parity.reset_index().to_dict('records'), quotes=records,
                  parity_iv_comparison=discrepancies,
                  invalid_iv_reason='Bid below discounted intrinsic lower bound; no nonnegative-volatility solution.')
    save_results(1, result)
    print(volatility.to_string(index=False))
    print('Flagged IV gap comparison:', discrepancies)


if __name__ == '__main__':
    main()
