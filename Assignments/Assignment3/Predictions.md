# Predictions recorded before the corresponding fits

These are initial expectations, not conclusions rewritten to match the results.
The questions explicitly allow parity checks in Problem 1, Greeks in Problem 3,
moments in Problem 4, and the initial CAPM in Problem 5 before making predictions.

## Problem 1

The parity check flags strike 175: the call-minus-put mid gap is about $0.888,
larger than the combined full spread of $0.47. At least one quote or contract
record is inconsistent; parity alone cannot identify which leg is wrong.
Exact parity implies equal call and put implied volatilities because GBSM
call-minus-put does not depend on volatility. Passing a spread tolerance does
not imply exact equality. Far-wing options should have low vega, so a cent of
rounding or spread produces a large IV change. The calls at 215, 220 and 225
all have the same $0.05/$0.10 quote despite different strikes: I expect an
unreliable upturn rather than three precise market volatility observations.

## Problem 2

The first dividend is below the strike-financing threshold; the second is
above it. I expect call exercise to be possible only just before dividend 2,
and only for a sufficiently high stock price. Dividends should lower both
calls and raise both puts. I expect the call exercise premium to increase and
the put exercise premium to decrease, because waiting captures the stock drop.
At a fixed number of tree steps, tiny-bump gamma should be unreliable because
the price is piecewise linear in the starting stock price.

## Problem 3

Initial book Greeks, ordered CoveredCall, ShortStraddle, RiskReversal,
HedgedShortCall: delta = 69.9161, -0.8888, 61.3887, -0.4444;
gamma = -3.7925, -8.0921, 0.5360, -4.0461; theta per year =
1477.7764, 3313.3549, 18.5884, 1868.1176.

I rank absolute dollar delta-normal misstatement as ShortStraddle,
HedgedShortCall, CoveredCall, then RiskReversal. The first two have almost
zero delta but material negative gamma, so delta-normal should understate
their risk. I also expect understatement for CoveredCall. RiskReversal's
small positive local gamma suggests a slight overstatement, but that local
approximation may not survive a large move. Positive starting theta predicts
positive unchanged-price P&L in all four books. I expect the short straddle
and hedged short call to have especially high ES/VaR ratios because large
moves hurt either side; a heavy-tailed return fit would increase the concern.

## Problem 4

PUTW has annualized Sharpe around 0.783, skewness around -6.57 and excess
kurtosis around 61.14 in the initial bias-corrected diagnostics. The final
reported central-moment convention gives -6.525 and 59.850, without changing
this prediction. One month supplies about 46% of its centered sum of
squares. I expect the long-only tangency portfolio to allocate over half to
PUTW, but its historical 1% ES to be appreciably worse than a normal with
the same mean and SD. This is a prediction about selling downside insurance,
not evidence of a better investment.

## Problem 5

The initial CAPM gives FUND1 a larger daily alpha (0.0004407, t=3.20) than
FUND2 (0.0002449, t=1.83). FUND1's CAPM residual correlation with momentum
is about 0.801, versus 0.053 for FUND2. I expect the momentum factor to
remove much of FUND1's apparent alpha, while FUND2's alpha changes less.
SMB and HML may also shift both intercepts.
