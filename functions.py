import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.diagnostic import het_breuschpagan


class OLS: # This class implements Ordinary Least Squares regression.
    def __init__(self, formula, data): # This method initializes the OLS class with a formula and data.
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        self.names = list(X.columns)
        self.y = y.values              
        self.X = X.values              
        self.n, self.k = self.X.shape

        self.beta = None
        self.sigmasq = None
        self.varbeta = None

    def fit(self): # This method fits the OLS model to the data.
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        
        self.beta = np.linalg.solve(XtX, Xty)

        uhat = self.y - self.X @ self.beta
        self.sigmasq = (uhat.T @ uhat).item() / (self.n - self.k)

        self.varbeta = self.sigmasq * np.linalg.inv(XtX)
        return self

    def snr(self): # This method calculates the signal-to-noise ratio (SNR) of the fitted model.
        self._check_fitted()
        ESS = (self.beta.T @ self.X.T @ self.X @ self.beta).item()
        RSS = self.sigmasq
        return ESS / RSS

    def summary(self): # This method returns a summary of the fitted model, including coefficients, standard errors, and t-statistics.
        self._check_fitted()
        se = np.sqrt(np.diag(self.varbeta))
        beta = self.beta.flatten()
        return pd.DataFrame(
            {"coef": beta, "std_err": se, "t": beta / se},
            index=self.names,
        )

    def _check_fitted(self): # This method checks if the model has been fitted by verifying if the beta coefficients are not None.
        if self.beta is None:
            raise RuntimeError("Call .fit() before using this method.")
    
    def r_squared(self): # This method calculates the R-squared value of the fitted model, which measures the proportion of variance in the dependent variable that is predictable from the independent variables.
        self._check_fitted()
        uhat = self.y - self.X @ self.beta
        RSS = (uhat.T @ uhat).item()

        if "Intercept" in self.names:
            TSS = ((self.y - self.y.mean()) ** 2).sum()   # centred
        else:
            TSS = (self.y.T @ self.y).item()              # uncentred

        return 1 - RSS / TSS

    def adj_r_squared(self): # This method calculates the adjusted R-squared value of the fitted model, which adjusts the R-squared value for the number of predictors in the model.
        r2 = self.r_squared()
        return 1 - (1 - r2) * (self.n - 1) / (self.n - self.k)
    
    def cov_beta(self): # This method returns the covariance matrix of the coefficient estimates, which is calculated as sigma^2 times the inverse of X'X.
        """Covariance matrix of the coefficient estimates: sigma^2 (X'X)^-1."""
        self._check_fitted()
        return pd.DataFrame(self.varbeta, index=self.names, columns=self.names)

    def corr_beta(self): # This method returns the correlation matrix of the coefficient estimates, which is calculated by normalizing the covariance matrix of the coefficient estimates.
        """Correlation matrix of the coefficient estimates."""
        self._check_fitted()
        se = np.sqrt(np.diag(self.varbeta))
        return pd.DataFrame(
            self.varbeta / np.outer(se, se),
            index=self.names, columns=self.names,
        )

    def cov_vars(self): # This method returns the sample covariance matrix of the regressors (excluding the intercept), which is calculated using the numpy covariance function on the design matrix X.
        """Sample covariance matrix of the regressors (intercept excluded)."""
        cols = [i for i, name in enumerate(self.names) if name != "Intercept"]
        Xr = self.X[:, cols]
        names = [self.names[i] for i in cols]
        return pd.DataFrame(np.cov(Xr, rowvar=False), index=names, columns=names)
    
    
class Test: # This class implements various statistical tests for the fitted OLS model.
    
    def __init__(self, model): # This method initializes the Test class with a fitted OLS model.
        self.model = model
        self.model._check_fitted()

    def heteroskedasticity_automatic(self): # This method performs the Breusch-Pagan test for heteroskedasticity on the fitted OLS model.
        """Breusch-Pagan test for heteroskedasticity."""
        uhat = self.model.y - self.model.X @ self.model.beta
        _, pval, _, _ = het_breuschpagan(uhat.flatten(), self.model.X)
        if pval < 0.05:
            print("Reject H0: Evidence of heteroskedasticity.")
        else:
            print("Fail to reject H0: No evidence of heteroskedasticity.")
            
        print("p-value:", pval)
            
        


    
