# OLS from Scratch

A small Python project that implements Ordinary Least Squares (OLS) regression from first principles using matrix algebra (NumPy), and tests it on a synthetic dataset with a known data-generating process.

## Overview

The project has two parts:

1. **Data simulation** – generate a dataset where the true coefficients are known.
2. **OLS estimator** – a custom `OLS` class that estimates the coefficients, standard errors, t-statistics, error variance and a signal-to-noise ratio, without relying on `statsmodels` or `scikit-learn`.

Because the true parameters are known, you can check how closely the estimator recovers them.

## Data-generating process

Data is produced by `generate_ols_dataset()` in `rand_data_analysis.py`:

```
Y = β0 + β1·X1 + β2·X2 + β3·X3 + ε
```

| Parameter | Default value |
|-----------|---------------|
| Observations (`n_obs`) | 1000 |
| Intercept (`beta_0`) | 5.0 |
| Slopes (`betas`) | 2.5, -1.8, 0.75 |
| Regressors | Uniform(-10, 10) |
| Noise ε | Normal(0, 2²) |

The output is saved to `data.csv` with columns `X1`, `X2`, `X3`, `error`, `Y`. Set `seed` for reproducible data, and a `.xlsx` path for Excel output.

## Estimator

The `OLS` class in `functions.py` takes an R-style formula (parsed with `patsy`) and a DataFrame.

- **Coefficients:** β̂ = (XᵀX)⁻¹Xᵀy, computed with `np.linalg.solve`
- **Error variance:** σ̂² = ûᵀû / (n − k)
- **Covariance matrix:** Var(β̂) = σ̂²(XᵀX)⁻¹
- **t-statistics:** β̂ / SE(β̂)
- **SNR:** ESS / σ̂², via `snr()`

| Method | Description |
|--------|-------------|
| `fit()` | Estimates β, σ̂² and Var(β̂). Returns `self` for chaining |
| `summary()` | DataFrame of coefficients, standard errors and t-statistics |
| `snr()` | Signal-to-noise ratio (requires `fit()` first) |

## Project structure

```
.
├── functions.py            # OLS class
├── rand_data_analysis.py   # Synthetic data generator
├── data.csv                # Generated dataset (1000 rows, 5 columns)
├── hello_git.py            # Git setup test script
└── README.md
```

## Getting started

### Requirements

- Python 3.9+
- `numpy`, `pandas`, `patsy`

```bash
pip install numpy pandas patsy
```

### Run the example

```bash
python functions.py
```

This generates a fresh dataset, fits `Y ~ X1 + X2 + X3`, and prints the summary table, σ̂² and the SNR.

### Use in your own code

```python
import pandas as pd
from functions import OLS

df = pd.read_csv("data.csv")
model = OLS("Y ~ X1 + X2 + X3", df).fit()

print(model.summary())
print("sigma^2:", model.sigmasq)
print("SNR:", model.snr())
```

### Generate custom data

```python
from rand_data_analysis import generate_ols_dataset

df = generate_ols_dataset(
    n_obs=5000,
    betas=[1.0, -0.5, 2.0],
    noise_std=1.0,
    seed=42,
    output_filepath="my_data.csv",
)
```

## Expected results

With the default settings, the estimates should be close to the true values (β0 = 5, β1 = 2.5, β2 = -1.8, β3 = 0.75) and σ̂² close to 4 (noise std = 2).

## Notes

- `functions.py` imports `generate_ols_dataset` from `rand_data_analysis.py`, so keep both files in the same directory.
- Running `functions.py` as a script regenerates and overwrites `data.csv` each time unless a different `output_filepath` is passed (or `None`).

## Possible extensions

- R², adjusted R² and F-statistic
- p-values and confidence intervals
- Heteroskedasticity-robust standard errors
- Unit tests comparing results against `statsmodels`

## Licence

Add a licence of your choice (e.g. MIT).
