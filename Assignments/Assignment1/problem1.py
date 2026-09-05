import numpy as np
import pandas as pd
from scipy.stats import norm


# Read the single data series from problem1.csv.
data = pd.read_csv("problem1.csv")
sample = data["x"].to_numpy()
number_of_observations = len(sample)

# Follow the Week 1 example to calculate the first four moments.
sample_mean = np.mean(sample)
centered_values = sample - sample_mean

# Week 1 uses n - 1 for the reported sample variance.
sample_variance = np.sum(centered_values**2) / (number_of_observations - 1)

# Week 1 uses central moments with n in the denominator for skewness and kurtosis.
central_variance = np.mean(centered_values**2)
sample_skewness = np.mean(centered_values**3) / central_variance**1.5
sample_excess_kurtosis = np.mean(centered_values**4) / central_variance**2 - 3

# Fit a Normal distribution by matching the sample mean and sample variance.
normal_standard_deviation = np.sqrt(sample_variance)
normal_one_percent_quantile = norm.ppf(
    0.01,
    loc=sample_mean,
    scale=normal_standard_deviation,
)

# Compare the observed lower-tail count with the count predicted by the Normal.
observed_below_quantile = np.sum(sample < normal_one_percent_quantile)
expected_below_quantile = 0.01 * number_of_observations

print(f"Number of observations: {number_of_observations}")
print(f"Mean: {sample_mean:.10f}")
print(f"Variance: {sample_variance:.10f}")
print(f"Skewness: {sample_skewness:.6f}")
print(f"Excess kurtosis: {sample_excess_kurtosis:.6f}")
print(f"Normal standard deviation: {normal_standard_deviation:.10f}")
print(f"Normal 1% quantile: {normal_one_percent_quantile:.10f}")
print(f"Observed below the 1% quantile: {observed_below_quantile}")
print(f"Expected below the 1% quantile: {expected_below_quantile:.1f}")
