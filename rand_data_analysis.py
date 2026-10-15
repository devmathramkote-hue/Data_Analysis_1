import pandas as pd
import numpy as np
import var

def generate_ols_dataset(
    n_obs=var.n_obs,
    beta_0=var.beta_0,
    betas=var.betas,
    x_bounds=var.x_bounds,
    noise_mean=var.noise_mean,
    noise_std=var.noise_std,
    include_error_col=var.include_error_col,
    seed=var.seed,
    output_filepath=var.output_filepath,
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