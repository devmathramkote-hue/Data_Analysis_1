from functions import OLS, Test
from rand_data_analysis import generate_ols_dataset


if __name__ == "__main__":
    df = generate_ols_dataset()

    regressors = [c for c in df.columns if c.startswith("X")]
    formula = "Y ~ " + " + ".join(regressors)
    
    model = OLS(formula, df).fit()
    
    print(model.summary())
    print("sigma^2:", model.sigmasq)
    print("SNR:", model.snr())
    print("R^2:", model.r_squared())
    print("Adj R^2:", model.adj_r_squared())
    
    #print(model.cov_beta())
    print(model.corr_beta())
    #print(model.cov_vars())
    print("")
    print("Heteroskedasticity Test:")
    test = Test(model)
    test.heteroskedasticity_automatic()