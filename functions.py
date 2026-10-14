import numpy as np
import pandas as pd
from patsy import dmatrices
from rand_data_anaylsis import generate_ols_dataset


class OLS:
    def __init__(self, formula, data):
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        self.names = list(X.columns)
        self.y = y.values              # shape (n, 1)
        self.X = X.values              # shape (n, k)
        self.n, self.k = self.X.shape

        self.beta = None
        self.sigmasq = None
        self.varbeta = None

    def fit(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        # Beta: solve is more stable than explicitly inverting
        self.beta = np.linalg.solve(XtX, Xty)

        # Residual variance
        uhat = self.y - self.X @ self.beta
        self.sigmasq = (uhat.T @ uhat).item() / (self.n - self.k)

        # Variance-covariance matrix of beta
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

'''
if __name__ == "__main__":
    df = generate_ols_dataset()
    model = OLS("y ~ x1 + x2", df).fit()   # adjust formula to your columns
    print(model.summary())
    print("sigma^2:", model.sigmasq)
    print("SNR:", model.snr())
'''