from functions import OLS
from rand_data_analysis import generate_ols_dataset


if __name__ == "__main__":
    
    
    df = generate_ols_dataset()
    model = OLS("Y ~ X1 + X2+X3", df).fit()   # adjust formula to your columns
    print(model.summary())
    print("sigma^2:", model.sigmasq)
    print("SNR:", model.snr())
    print("R^2:", model.r_squared())
    print("Adj R^2:", model.adj_r_squared())