from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


# Read the time series.
data = pd.read_csv("problem5.csv")
series = data["x"].to_numpy()
number_of_observations = len(series)
significance_band = 1.96 / np.sqrt(number_of_observations)

# Calculate the ACF and PACF values used for the prediction.
acf_values = acf(series, nlags=20, fft=False)
pacf_values = pacf(series, nlags=20, method="ywm")

# Plot the series, ACF, and PACF before fitting the models.
figure_folder = Path("figures")
figure_folder.mkdir(exist_ok=True)
figure, axes = plt.subplots(3, 1, figsize=(8, 8))

axes[0].plot(range(1, number_of_observations + 1), series, color="#2F5D8A")
axes[0].set_title("Series x")
axes[0].set_xlabel("Observation")
axes[0].set_ylabel("x")
axes[0].grid(alpha=0.2)

plot_acf(
    series,
    lags=20,
    alpha=0.05,
    zero=False,
    bartlett_confint=False,
    ax=axes[1],
    color="#2F5D8A",
    vlines_kwargs={"colors": "#2F5D8A"},
)
axes[1].set_title("ACF with 95% significance band")

plot_pacf(
    series,
    lags=20,
    alpha=0.05,
    zero=False,
    method="ywm",
    ax=axes[2],
    color="#2F5D8A",
    vlines_kwargs={"colors": "#2F5D8A"},
)
axes[2].set_title("PACF with 95% significance band")

figure.tight_layout()
figure.savefig(figure_folder / "problem5_acf_pacf.png", dpi=150)
plt.close(figure)

# Fit the six candidate models.
model_orders = {
    "AR(1)": (1, 0, 0),
    "AR(2)": (2, 0, 0),
    "AR(3)": (3, 0, 0),
    "MA(1)": (0, 0, 1),
    "MA(2)": (0, 0, 2),
    "MA(3)": (0, 0, 3),
}

fitted_models = {}
for model_name, model_order in model_orders.items():
    fitted_models[model_name] = ARIMA(series, order=model_order, trend="c").fit()

print(f"95% significance band: +/-{significance_band:.6f}")
for lag in range(1, 6):
    print(
        f"Lag {lag}: ACF {acf_values[lag]:.6f}, "
        f"PACF {pacf_values[lag]:.6f}"
    )

print("AICc values")
for model_name, fitted_model in fitted_models.items():
    print(f"  {model_name}: {fitted_model.aicc:.3f}")

selected_model = min(fitted_models, key=lambda name: fitted_models[name].aicc)
print(f"Selected model: {selected_model}")

ar2_model = fitted_models["AR(2)"]
ar3_model = fitted_models["AR(3)"]
print(f"AR(2) coefficients: {ar2_model.params[1]:.6f}, {ar2_model.params[2]:.6f}")
print(
    "AR(3) coefficients: "
    f"{ar3_model.params[1]:.6f}, {ar3_model.params[2]:.6f}, "
    f"{ar3_model.params[3]:.6f}"
)
