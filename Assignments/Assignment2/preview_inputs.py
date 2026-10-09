"""Make the required raw-data plots BEFORE fitting any models."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from helpers import FOLDER


def main():
    figures = FOLDER / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    prices = pd.read_csv(FOLDER / "problem2.csv")
    returns = prices.Price.pct_change().dropna()
    returns = returns - returns.mean()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(prices.Day.iloc[1:], returns * 100, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set(xlabel="Day", ylabel="Demeaned return (%)",
           title="Problem 2: the recent volatility regime looks different")
    fig.tight_layout()
    fig.savefig(figures / "problem2_returns.png", dpi=180)
    plt.close(fig)

    bonds = pd.read_csv(FOLDER / "problem3.csv")
    positions = {"$2 million in A": 2_000_000 * bonds.A,
                 "$1 million in each": 1_000_000 * (bonds.A + bonds.B)}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True, sharey=True)
    for ax, (name, pnl) in zip(axes, positions.items()):
        ax.hist(pnl / 1000, bins=70, color="#477c9e", edgecolor="white")
        ax.set(title=name, xlabel="P&L ($ thousands)")
        ax.set_xticks([-1500, -1000, -500, 0, 250])
        ax.axvline(0, color="gray", linewidth=0.6)
    axes[0].set_ylabel("Scenarios (log scale)")
    axes[0].set_yscale("log")  # Makes the infrequent defaults visible too.
    fig.tight_layout()
    fig.savefig(figures / "problem3_pnl.png", dpi=180)
    plt.close(fig)

    assets = pd.read_csv(FOLDER / "problem4.csv")
    uniforms = assets.rank() / (len(assets) + 1)
    pairs = [("X1", "X2"), ("X1", "X3"), ("X2", "X3")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (first, second) in zip(axes, pairs):
        ax.scatter(uniforms[first], uniforms[second], s=7, alpha=0.4)
        for cutoff in [0.025, 0.975]:
            ax.axhline(cutoff, color="gray", linewidth=0.6, linestyle="--")
            ax.axvline(cutoff, color="gray", linewidth=0.6, linestyle="--")
        ax.set(xlabel=first + " rank / (n + 1)", ylabel=second + " rank / (n + 1)",
               xlim=(0, 1), ylim=(0, 1))
    fig.suptitle("Problem 4: rank plots before margin or copula fitting")
    fig.tight_layout()
    fig.savefig(figures / "problem4_ranks.png", dpi=180)
    plt.close(fig)
    print("Saved the three raw-data figures in", figures)


if __name__ == "__main__":
    main()
