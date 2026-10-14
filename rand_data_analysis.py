import pandas as pd
import numpy as np

def generate_ols_dataset(
    n_obs=1000,
    beta_0=5.0,
    betas=[2.5, -1.8, 0.75],
    x_bounds=(-10.0, 10.0),
    noise_mean=0.0,
    noise_std=2.0,
    include_error_col=True,
    seed=None,
    output_filepath='data.csv',
):
    rng = np.random.default_rng(seed)
    betas = np.asarray(betas)
    k = len(betas)

    # n_obs x k matrix of uniform features
    X = rng.uniform(x_bounds[0], x_bounds[1], size=(n_obs, k))

    # Gaussian errors
    error = rng.normal(noise_mean, noise_std, size=n_obs)

    # Y = beta_0 + X @ betas + error
    y = beta_0 + X @ betas + error

    df = pd.DataFrame(X, columns=[f"X{j+1}" for j in range(k)])
    if include_error_col:
        df["error"] = error
    df["Y"] = y

    if output_filepath:
        if output_filepath.endswith((".xlsx", ".xls")):
            df.to_excel(output_filepath, index=False)
        else:
            df.to_csv(output_filepath, index=False)

    return df