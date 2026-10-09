# Predictions recorded before model fitting

These predictions use only missing-data counts, raw moments, and the figures
created by `preview_inputs.py`. They are kept separate so that an incorrect
prediction is not silently replaced after seeing the fitted results.

## Problem 1

Only 81 days contain all five returns. D has the shortest history, so correlations
involving D should be least reliable. Complete-case correlation is PSD because it
uses one common set of observations; pairwise correlation need not be. The index
is almost a weighted copy of the stocks, so its smallest eigenvalue should be
close to zero. Small pairwise estimation differences could make it negative.

## Problem 2

The last roughly 40 returns look more volatile than the first 460. My provisional
smallest-to-largest 5% VaR ranking is historical, fitted t, equally weighted normal,
EW normal at 0.97, EW normal at 0.94. The first three are the uncertain part: they
mix a long quiet period with a short volatile period. EW weights should react to
the new regime, and 0.94 should react most quickly. Full-sample excess kurtosis
of 2.132 is hard to reconcile with one constant-volatility normal regime.

## Problem 3

A has 406 large-loss scenarios, B has 393, and their union has 782 out of 10,000.
Each individual default frequency is below 5%, but the union is above 5%.
Diversifying should therefore raise 5% VaR, because its quantile enters the
default-loss cluster. It should lower ES relative to putting $2 million in A:
most diversified defaults affect only half the investment. The separated gain
and default-loss clusters make a single normal distribution unsuitable.

## Problem 4

X1 looks normal; X2 and X3 look heavy-tailed, so I would try t margins for them.
X2's excess kurtosis falls from 7.083 to 1.710 when its most extreme observation
is removed, but some heavy-tailed behavior remains. Rank plots show shared
extremes. Joint lower/upper 2.5% counts are X1/X2: 6/8, X1/X3: 7/12, X2/X3: 6/9,
compared with 0.625 under independence. I expect the t copula to win; positive
ordinary correlation alone can also produce more joint extremes than independence.

## Problem 5

If the stocks have positive residual covariance, removing it deletes the
positive cross term in the long-long portfolio variance and the negative cross
term in the long-short variance. The shortcut should understate P1 VaR and
overstate P2 VaR. This prediction is conditional on that positive covariance;
the regressions will determine whether the data support it.
