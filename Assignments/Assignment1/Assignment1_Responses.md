# Assignment 1 - Univariate and Multivariate Statistics

## Problem 1: Reading the Shape of a Sample

### Results

Using the Week 1 conventions, the sample has 1,000 observations:

- Mean: 0.0011043943
- Variance: 0.0000962482
- Skewness: -0.668831
- Excess kurtosis: 2.339849

The variance uses `n - 1`. Kurtosis is reported as excess kurtosis.

### (a) Predict

The negative skewness and positive excess kurtosis make NIG plausible. Normal is
ruled out by both moments. Lognormal is ruled out because its skewness should be
positive. Student's t is ruled out because it is symmetric and should have zero
skewness.

### (b) Fit

The fitted Normal has mean 0.0011043943, variance 0.0000962482, and standard
deviation 0.0098106178. Its 1% quantile is -0.0217185155.

- Observed below the 1% quantile: 26
- Expected under the Normal: 10

### (c) Reconcile

The Normal underestimates left-tail risk. It predicts 1% below the cutoff, but
the sample has 2.6%. The sample is more left-skewed and heavy-tailed than the
fitted Normal.

\newpage

## Problem 2: A Regression Whose Errors Are Not Normal

### (a)-(b) Predict

The scatter is strongly linear but has several large vertical outliers, so I
expect Student's t errors. This violates OLS assumption 7: Normal errors. The
slope should remain unbiased and consistent if assumptions 1-6 hold, but heavy
tails make it less efficient and more sensitive to outliers. I expect its
estimated standard error to be larger and less stable, and exact Normal-based
tests and intervals no longer apply.

### Fit

- OLS: alpha 1.560740 (SE 0.096269), beta 2.990757 (SE 0.096640), residual standard deviation 1.351401
- Normal MLE: alpha 1.560740, beta 2.990757, scale 1.344627, AICc 692.145
- Student's t MLE: alpha 1.547980, beta 3.019179, scale 0.935509, degrees of freedom 3.632160, AICc 665.327

### (c)-(e) Reconcile

The three slopes are close, so non-Normal errors did not move the slope much.
Student's t wins by 26.817 AICc points because the Normal misses the heavy error
tails, not because it misses the slope.

- 95% error quantile: Normal 2.211715; Student's t 2.053833
- 99.5% error quantile: Normal 3.463530; Student's t 4.620089

The Normal is wider at 95% because its fitted scale is larger, but the t is wider
in the far tail. I would use the t model for a capital buffer.

\newpage

## Problem 3: Pearson Against Spearman

### (a) Predict

I expect the largest disagreement for x1 and x2 because their relationship is
strongly monotonic but curved. Pearson and Spearman should be close for the
roughly linear x1-x3 pair and for the pairs involving independent x4.

### (b) Fit

Pearson correlation matrix:

```text
       x1        x2        x3        x4
x1  1.000000  0.799092  0.719467 -0.004680
x2  0.799092  1.000000  0.587614 -0.020923
x3  0.719467  0.587614  1.000000 -0.008605
x4 -0.004680 -0.020923 -0.008605  1.000000
```

Spearman correlation matrix:

```text
       x1        x2        x3        x4
x1  1.000000  0.971930  0.683910  0.000189
x2  0.971930  1.000000  0.667422 -0.001131
x3  0.683910  0.667422  1.000000 -0.002422
x4  0.000189 -0.001131 -0.002422  1.000000
```

The largest gap is x1-x2: Pearson 0.799092, Spearman 0.971930, gap 0.172838.

### (c) Reconcile

The curve keeps almost the same ranking but is not a straight line. Spearman is
the clearer description of its monotonic strength. Pearson answers the narrower
question of how strongly the pair follows a straight line.

\newpage

## Problem 4: Conditional Distributions

To follow the Week 2 notation, block X1 is the target series x2 and block X2 is
the observed series x1. Therefore, Sigma11 = Var(x2) = 4.104749, Sigma22 =
Var(x1) = 1.099875, and Sigma12 = Sigma21 = Cov(x2,x1) = 1.696902.

### (a)-(b) Predict

`Var(x2 | x1) = Sigma11 - Sigma12 * Sigma22^-1 * Sigma21 = 1.486746`.
The remaining variance factor is
`(Sigma11 - Sigma12 * Sigma22^-1 * Sigma21) / Sigma11 = 0.362201`, a 63.78%
reduction. The remaining standard-deviation factor is 0.601832. Under the
multivariate Normal, this does not depend on observed x1 because the variance
formula contains no observed value.

### (c) Fit

`E[x2 | x1=a] = mu1 + Sigma12 * Sigma22^-1 * (a - mu2)`, where mu1 is
mean(x2) and mu2 is mean(x1).
The coefficient on x1 is 1.542813, which is the OLS slope. The equivalent line
is 0.071073 + 1.542813a. The constant 95% band has half-width 2.389827.

### (d)-(f) Reconcile

- Overall coverage: 93.50%
- Within 1 SD of mean x1: 95.65% (759 observations)
- Between 1 and 2 SD: 85.79% (190 observations)
- Beyond 2 SD: 90.20% (51 observations)

Coverage is not flat, so the conditional variance is not constant. The OLS
slope remains the best linear coefficient, but the exact Normal conditional
distribution and constant-width 95% band do not survive this failure.

\newpage

## Problem 5: Identifying an AR or MA Order

### (a)-(b) Predict

The ACF decays with an oscillating pattern. The PACF has large values at lags 1
and 2, then mostly cuts off. Using the approximate 95% band of +/-0.087654, I
predict AR(2). The rule is: an AR(p) has a decaying ACF and a PACF that cuts off
after lag p.

### (c)-(d) Fit and reconcile

- AR(1): AICc 1418.587
- AR(2): AICc 1372.438
- AR(3): AICc 1374.343
- MA(1): AICc 1389.084
- MA(2): AICc 1381.232
- MA(3): AICc 1374.117

AICc selects AR(2), matching the prediction.

### (e) AR(2) versus AR(3)

AR(2) coefficients are 0.601201 and -0.302638. AR(3) coefficients are 0.596203,
-0.292690, and -0.016542. The third coefficient is very small and barely
improves fit, so AICc rejects it after penalizing the extra parameter. R-squared
does not penalize added variables and therefore would not make the same choice.
