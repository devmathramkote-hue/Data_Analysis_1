import numpy as np
import pandas as pd
from patsy import dmatrices
from rand_data_analysis import generate_ols_dataset


class OLS:
    def __init__(self, formula, data):
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        self.names = list(X.columns)
        self.y = y.values              
        self.X = X.values              
        self.n, self.k = self.X.shape

        self.beta = None
        self.sigmasq = None
        self.varbeta = None

    def fit(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        
        self.beta = np.linalg.solve(XtX, Xty)

        uhat = self.y - self.X @ self.beta
        self.sigmasq = (uhat.T @ uhat).item() / (self.n - self.k)

        self.varbeta = self.sigmasq * np.linalg.inv(XtX)
        return self

    def snr(self):
        self._check_fitted()
        ESS = (self.beta.T @ self.X.T @ self.X @ self.beta).item()
        RSS = self.sigmasq
        return ESS / RSS

    def summary(self):
        self._check_fitted()
        se = np.sqrt(np.diag(self.varbeta))
        beta = self.beta.flatten()
        return pd.DataFrame(
            {"coef": beta, "std_err": se, "t": beta / se},
            index=self.names,
        )

    def _check_fitted(self):
        if self.beta is None:
            raise RuntimeError("Call .fit() before using this method.")
    
    def r_squared(self):
        self._check_fitted()
        uhat = self.y - self.X @ self.beta
        RSS = (uhat.T @ uhat).item()

        if "Intercept" in self.names:
            TSS = ((self.y - self.y.mean()) ** 2).sum()   # centred
        else:
            TSS = (self.y.T @ self.y).item()              # uncentred

        return 1 - RSS / TSS

    def adj_r_squared(self):
        r2 = self.r_squared()
        return 1 - (1 - r2) * (self.n - 1) / (self.n - self.k)


if __name__ == "__main__":
    
    
    df = generate_ols_dataset()
    model = OLS("Y ~ X1 + X2+X3", df).fit()   # adjust formula to your columns
    print(model.summary())
    print("sigma^2:", model.sigmasq)
    print("SNR:", model.snr())
    print("R^2:", model.r_squared())
    print("Adj R^2:", model.adj_r_squared())
    
