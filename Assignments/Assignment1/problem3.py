from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Read the four series.
data = pd.read_csv("problem3.csv")

# Plot every pair, following the Week 2 correlation examples.
figure_folder = Path("figures")
figure_folder.mkdir(exist_ok=True)
series_pairs = list(combinations(data.columns, 2))
figure, axes = plt.subplots(2, 3, figsize=(10, 6.5))

for axis, (first_series, second_series) in zip(axes.flat, series_pairs):
    axis.scatter(
        data[first_series],
        data[second_series],
        s=14,
        alpha=0.6,
        color="#2F5D8A",
    )
    axis.set_title(f"{first_series} vs {second_series}")
    axis.set_xlabel(first_series)
    axis.set_ylabel(second_series)
    axis.grid(alpha=0.2)

figure.suptitle("Problem 3: Pairwise scatter plots")
figure.tight_layout()
figure.savefig(figure_folder / "problem3_pairs.png", dpi=150)
plt.close(figure)

# After looking at the plots, calculate both correlation matrices.
pearson_correlation = data.corr(method="pearson")
spearman_correlation = data.corr(method="spearman")

# Find the pair with the largest absolute difference.
largest_gap = 0.0
largest_gap_pair = None

for first_series, second_series in series_pairs:
    gap = abs(
        pearson_correlation.loc[first_series, second_series]
        - spearman_correlation.loc[first_series, second_series]
    )
    if gap > largest_gap:
        largest_gap = gap
        largest_gap_pair = (first_series, second_series)

first_series, second_series = largest_gap_pair
print("Pearson correlation matrix")
print(pearson_correlation.round(6))
print("\nSpearman correlation matrix")
print(spearman_correlation.round(6))
print(f"\nLargest gap: {first_series} and {second_series}")
print(f"Pearson: {pearson_correlation.loc[first_series, second_series]:.6f}")
print(f"Spearman: {spearman_correlation.loc[first_series, second_series]:.6f}")
print(f"Absolute gap: {largest_gap:.6f}")
