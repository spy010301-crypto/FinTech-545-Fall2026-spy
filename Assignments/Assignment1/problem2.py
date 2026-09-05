from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, t


# Read the data.
data = pd.read_csv("problem2.csv")
x_values = data["x"].to_numpy()
y_values = data["y"].to_numpy()
number_of_observations = len(data)

# Plot the data before fitting any model.
figure_folder = Path("figures")
figure_folder.mkdir(exist_ok=True)

figure, axis = plt.subplots(figsize=(7, 4.5))
axis.scatter(x_values, y_values, s=18, alpha=0.65, color="#2F5D8A")
axis.set_title("Problem 2: y versus x", pad=12)
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.grid(alpha=0.2)
figure.subplots_adjust(top=0.88, left=0.11, right=0.97, bottom=0.13)
figure.savefig(figure_folder / "problem2_scatter.png", dpi=150)
plt.close(figure)

# Fit OLS. Adding a column of ones gives the model an intercept.
design_matrix = sm.add_constant(x_values)
ols_model = sm.OLS(y_values, design_matrix).fit()
ols_alpha, ols_beta = ols_model.params
ols_alpha_se, ols_beta_se = ols_model.bse
ols_residuals = ols_model.resid
ols_residual_standard_deviation = np.sqrt(
    np.sum(ols_residuals**2) / (number_of_observations - 2)
)


def corrected_aic(log_likelihood, number_of_parameters):
    """Calculate AICc from a model's log likelihood."""
    aic = 2 * number_of_parameters - 2 * log_likelihood
    correction = (
        2
        * number_of_parameters
        * (number_of_parameters + 1)
        / (number_of_observations - number_of_parameters - 1)
    )
    return aic + correction


# Under Normal errors, the MLE regression coefficients equal the OLS coefficients.
normal_alpha = ols_alpha
normal_beta = ols_beta
normal_scale = np.sqrt(np.mean(ols_residuals**2))
normal_log_likelihood = np.sum(norm.logpdf(ols_residuals, scale=normal_scale))
normal_aicc = corrected_aic(normal_log_likelihood, 3)


# Fit the same line with Student's t errors.
def negative_t_log_likelihood(parameters):
    alpha, beta, log_scale, log_degrees_of_freedom = parameters
    scale = np.exp(log_scale)
    degrees_of_freedom = np.exp(log_degrees_of_freedom)
    residuals = y_values - (alpha + beta * x_values)
    return -np.sum(t.logpdf(residuals, df=degrees_of_freedom, scale=scale))


starting_values = [ols_alpha, ols_beta, np.log(normal_scale), np.log(5.0)]
t_fit = minimize(negative_t_log_likelihood, starting_values, method="L-BFGS-B")

t_alpha, t_beta, log_t_scale, log_t_degrees_of_freedom = t_fit.x
t_scale = np.exp(log_t_scale)
t_degrees_of_freedom = np.exp(log_t_degrees_of_freedom)
t_log_likelihood = -t_fit.fun
t_aicc = corrected_aic(t_log_likelihood, 4)

# Compare moderate and far-tail error quantiles.
normal_95_quantile = norm.ppf(0.95, scale=normal_scale)
t_95_quantile = t.ppf(0.95, df=t_degrees_of_freedom, scale=t_scale)
normal_995_quantile = norm.ppf(0.995, scale=normal_scale)
t_995_quantile = t.ppf(0.995, df=t_degrees_of_freedom, scale=t_scale)

print("OLS")
print(f"  alpha: {ols_alpha:.6f} (SE {ols_alpha_se:.6f})")
print(f"  beta: {ols_beta:.6f} (SE {ols_beta_se:.6f})")
print(f"  residual standard deviation: {ols_residual_standard_deviation:.6f}")

print("Normal MLE")
print(f"  alpha: {normal_alpha:.6f}")
print(f"  beta: {normal_beta:.6f}")
print(f"  scale: {normal_scale:.6f}")
print(f"  AICc: {normal_aicc:.3f}")

print("Student's t MLE")
print(f"  alpha: {t_alpha:.6f}")
print(f"  beta: {t_beta:.6f}")
print(f"  scale: {t_scale:.6f}")
print(f"  degrees of freedom: {t_degrees_of_freedom:.6f}")
print(f"  AICc: {t_aicc:.3f}")

print("Error quantiles")
print(f"  95%: Normal {normal_95_quantile:.6f}, t {t_95_quantile:.6f}")
print(f"  99.5%: Normal {normal_995_quantile:.6f}, t {t_995_quantile:.6f}")
