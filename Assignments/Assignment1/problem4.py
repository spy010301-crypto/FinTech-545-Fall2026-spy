from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


# Read the two correlated series.
data = pd.read_csv("problem4.csv")
x1_values = data["x1"].to_numpy()
x2_values = data["x2"].to_numpy()

sample_means = data.mean()
covariance_matrix = data.cov()

mean_x1 = sample_means["x1"]
mean_x2 = sample_means["x2"]
variance_x1 = covariance_matrix.loc["x1", "x1"]
variance_x2 = covariance_matrix.loc["x2", "x2"]
covariance_x1_x2 = covariance_matrix.loc["x1", "x2"]

# In Week 2's notation, target x2 is block X1 and observed x1 is block X2.
# The named variables below keep the Python formula easy to read.
conditional_variance = variance_x2 - covariance_x1_x2**2 / variance_x1
variance_remaining_factor = conditional_variance / variance_x2
standard_deviation_remaining_factor = np.sqrt(variance_remaining_factor)

regression_slope = covariance_x1_x2 / variance_x1
regression_intercept = mean_x2 - regression_slope * mean_x1
conditional_mean = mean_x2 + regression_slope * (x1_values - mean_x1)

# Build the constant-width 95% Normal band.
conditional_standard_deviation = np.sqrt(conditional_variance)
band_half_width = norm.ppf(0.975) * conditional_standard_deviation
lower_band = conditional_mean - band_half_width
upper_band = conditional_mean + band_half_width
inside_band = (x2_values >= lower_band) & (x2_values <= upper_band)

# Split observations by the distance of x1 from its mean.
distance_from_mean = abs(x1_values - mean_x1) / np.sqrt(variance_x1)
inside_one_standard_deviation = distance_from_mean <= 1
between_one_and_two = (distance_from_mean > 1) & (distance_from_mean <= 2)
beyond_two = distance_from_mean > 2

# Plot the data, conditional mean, and 95% band.
figure_folder = Path("figures")
figure_folder.mkdir(exist_ok=True)
sorted_positions = np.argsort(x1_values)

plt.figure(figsize=(7, 5))
plt.scatter(
    x1_values,
    x2_values,
    s=14,
    alpha=0.55,
    color="#6F8FAF",
    label="Data",
)
plt.plot(
    x1_values[sorted_positions],
    conditional_mean[sorted_positions],
    color="#17365D",
    linewidth=2,
    label="Conditional mean",
)
plt.fill_between(
    x1_values[sorted_positions],
    lower_band[sorted_positions],
    upper_band[sorted_positions],
    color="#D8E4F0",
    alpha=0.8,
    label="95% Normal band",
)
plt.title("Problem 4: Conditional mean and 95% band")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(figure_folder / "problem4_conditional.png", dpi=150)
plt.close()

print("Sample covariance matrix")
print(covariance_matrix.round(6))
print(f"Conditional variance: {conditional_variance:.6f}")
print(f"Variance remaining factor: {variance_remaining_factor:.6f}")
print(
    "Standard deviation remaining factor: "
    f"{standard_deviation_remaining_factor:.6f}"
)
print(f"Regression intercept: {regression_intercept:.6f}")
print(f"Regression slope: {regression_slope:.6f}")
print(f"95% band half-width: {band_half_width:.6f}")
print(f"Overall coverage: {inside_band.mean():.4%}")
print(
    "Coverage inside 1 SD: "
    f"{inside_band[inside_one_standard_deviation].mean():.4%} "
    f"({inside_one_standard_deviation.sum()} observations)"
)
print(
    "Coverage between 1 and 2 SD: "
    f"{inside_band[between_one_and_two].mean():.4%} "
    f"({between_one_and_two.sum()} observations)"
)
print(
    "Coverage beyond 2 SD: "
    f"{inside_band[beyond_two].mean():.4%} "
    f"({beyond_two.sum()} observations)"
)
