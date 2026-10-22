# OLS from Scratch

A small Python project that implements Ordinary Least Squares (OLS) regression from first principles using matrix algebra (NumPy), together with diagnostic tests, and evaluates it on a synthetic dataset with a known data-generating process.

## Overview

The project has four parts:

1. **Configuration** – a single `config.py` holds every simulation parameter.
2. **Data simulation** – generate a dataset where the true coefficients are known.
3. **OLS estimator** – a custom `OLS` class that estimates coefficients, standard errors, t-statistics, error variance, goodness of fit and coefficient covariance/correlation, without relying on `statsmodels` or `scikit-learn` for estimation.
4. **Diagnostic tests** – a `Test` class implementing the Breusch-Pagan test for heteroskedasticity, both via `statsmodels` and manually.

Because the true parameters are known, you can check how closely the estimator recovers them.

## Data-generating process

Data is produced by `generate_ols_dataset()` in `rand_data_analysis.py`:

```
Y = β0 + β1·X1 + β2·X2 + ... + βk·Xk + ε
```

The number of regressors is set by the length of `betas`. All defaults come from `config.py`:

| Parameter (`config.py`) | Default value | Description |
|-------------------------|---------------|-------------|
| `n_obs` | 1000 | Number of observations |
| `beta_0` | 50 | Intercept |
| `betas` | `[7, 1.57]` | Slope coefficients (two regressors: X1, X2) |
| `x_bounds` | `(-10.0, 10.0)` | Regressors drawn from Uniform(low, high) |
| `noise_mean` | 5.0 | Mean of the error term |
| `noise_std` | 4.0 | Standard deviation of the error term |
| `include_error_col` | `True` | Include the true error `ε` as a column |
| `seed` | 44 | Random seed for reproducible data |
| `output_filepath` | `'data.csv'` | Output path (`.csv`, or `.xlsx`/`.xls` for Excel; `None` to skip saving) |

The output DataFrame has columns `X1, X2, ..., error, Y`. Any argument to `generate_ols_dataset()` overrides the corresponding `config.py` value.

> **Note:** because the noise has a non-zero mean (`noise_mean = 5.0`), the estimated intercept absorbs it and converges to `beta_0 + noise_mean` (55), not `beta_0` (50).

## Estimator

The `OLS` class in `functions.py` takes an R-style formula (parsed with `patsy`) and a DataFrame.

- **Coefficients:** β̂ = (XᵀX)⁻¹Xᵀy, computed with `np.linalg.solve`
- **Error variance:** σ̂² = ûᵀû / (n − k)
- **Covariance matrix:** Var(β̂) = σ̂²(XᵀX)⁻¹
- **t-statistics:** β̂ / SE(β̂)
- **R²:** 1 − RSS/TSS, centred if the model has an intercept and uncentred otherwise
- **Adjusted R²:** 1 − (1 − R²)(n − 1)/(n − k)
- **SNR:** ESS / σ̂²

| Method | Description |
|--------|-------------|
| `fit()` | Estimates β, σ̂² and Var(β̂). Returns `self` for chaining |
| `summary()` | DataFrame of coefficients, standard errors and t-statistics |
| `snr()` | Signal-to-noise ratio |
| `r_squared()` | R² (centred or uncentred depending on the intercept) |
| `adj_r_squared()` | Adjusted R² |
| `cov_beta()` | Covariance matrix of the coefficient estimates, σ̂²(XᵀX)⁻¹ |
| `corr_beta()` | Correlation matrix of the coefficient estimates |
| `cov_vars()` | Sample covariance matrix of the regressors (intercept excluded) |

All methods other than `fit()` raise a `RuntimeError` if the model has not been fitted.

## Diagnostic tests

The `Test` class in `functions.py` wraps a fitted `OLS` model (it raises if the model is not fitted yet).

| Method | Description |
|--------|-------------|
| `heteroskedasticity_automatic()` | Breusch-Pagan test using `statsmodels.stats.diagnostic.het_breuschpagan` |
| `heteroskedasticity_manual()` | Breusch-Pagan LM test built from scratch: regress û² on the regressors using the `OLS` class, then LM = n·R² ~ χ²(k − 1) |

Both print the decision at the 5% level (reject or fail to reject H0 of homoskedasticity) and the p-value. The manual version is a useful cross-check against the `statsmodels` result. Since the simulated errors are homoskedastic, both tests should usually fail to reject H0.

## Project structure

```
.
├── config.py               # Simulation parameters (defaults)
├── rand_data_analysis.py   # Synthetic data generator
├── functions.py            # OLS and Test classes
├── main.py                 # Example run: simulate, fit, summarise, test
├── data.csv                # Generated dataset (1000 rows)
├── hello_git.py            # Git setup test script
└── README.md
```

## Getting started

### Requirements

- Python 3.9+
- `numpy`, `pandas`, `patsy`, `scipy`, `statsmodels`
- `openpyxl` (only if writing `.xlsx` output)

```bash
pip install numpy pandas patsy scipy statsmodels
```

### Run the example

```bash
python main.py
```

This generates a fresh dataset, builds the formula from every column starting with `X` (here `Y ~ X1 + X2`), fits the model and prints:

- the summary table (coefficients, standard errors, t-statistics)
- σ̂², SNR, R² and adjusted R²
- the covariance matrix of the coefficient estimates
- the Breusch-Pagan heteroskedasticity test result and p-value

Uncomment `model.corr_beta()` or `model.cov_vars()` in `main.py` to print those as well.

### Use in your own code

```python
import pandas as pd
from functions import OLS, Test

df = pd.read_csv("data.csv")
model = OLS("Y ~ X1 + X2", df).fit()

print(model.summary())
print("sigma^2:", model.sigmasq)
print("R^2:", model.r_squared())
print("Adj R^2:", model.adj_r_squared())

Test(model).heteroskedasticity_automatic()
```

### Generate custom data

```python
from rand_data_analysis import generate_ols_dataset

df = generate_ols_dataset(
    n_obs=5000,
    beta_0=1.0,
    betas=[1.0, -0.5, 2.0],
    noise_mean=0.0,
    noise_std=1.0,
    seed=42,
    output_filepath="my_data.csv",
)
```

## Expected results

With the default `config.py` settings, the estimates should be close to:

| Quantity | Expected value |
|----------|----------------|
| Intercept | ≈ 55 (`beta_0` + `noise_mean`) |
| β1 | ≈ 7 |
| β2 | ≈ 1.57 |
| σ̂² | ≈ 16 (`noise_std`² = 4²) |

## Notes

- `main.py` calls `generate_ols_dataset()` itself; `functions.py` no longer depends on the data generator, so it can be imported on its own.
- `rand_data_analysis.py` reads its defaults from `config.py` at import time, so edit `config.py` (or pass arguments explicitly) to change the simulation.
- Running `main.py` regenerates and overwrites `data.csv` each time unless a different `output_filepath` is passed (or `None`).
- Regressor columns must start with `X` for `main.py` to include them in the formula automatically; the `error` column is deliberately excluded.

## Possible extensions

- F-statistic, p-values and confidence intervals
- Heteroskedasticity-robust (White/HC) standard errors
- Further diagnostics (White test, Durbin-Watson, Jarque-Bera)
- Unit tests comparing results against `statsmodels`

## Licence

Add a licence of your choice (e.g. MIT).