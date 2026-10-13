import random as rand
import pandas as pd




def generate_ols_dataset(
    n_obs=1000,
    beta_0=5.0,
    betas=[2.5, -1.8, 0.75],
    x_bounds=(-10.0, 10.0),
    noise_mean=0.0,
    noise_std=2.0,
    include_error_col=True,
    seed=None,
    output_filepath=None,
):
    if seed is not None:
        rand.seed(seed)

    data = []
    num_features = len(betas)

    for _ in range(n_obs):
        # 1. Generate independent features X_1 ... X_k
        row_x = [rand.uniform(x_bounds[0], x_bounds[1]) for _ in range(num_features)]

        # 2. Draw random Gaussian error term (epsilon)
        error = rand.gauss(noise_mean, noise_std)

        # 3. Compute true Y = beta_0 + sum(beta_i * X_i) + error
        y = beta_0 + sum(b * x for b, x in zip(betas, row_x)) + error

        # 4. Assemble row dictionary
        record = {f"X{j+1}": row_x[j] for j in range(num_features)}
        if include_error_col:
            record["error"] = error
        record["Y"] = y

        data.append(record)

    df = pd.DataFrame(data)

    # 5. Optional file export
    if output_filepath:
        if output_filepath.endswith(".csv"):
            df.to_csv(output_filepath, index=False)
        elif output_filepath.endswith((".xlsx", ".xls")):
            df.to_excel(output_filepath, index=False)
        else:
            # Default fallback export
            df.to_csv(output_filepath, index=False)

    return df

