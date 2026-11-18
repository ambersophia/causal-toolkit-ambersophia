import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression, LinearRegression


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """Estimate the ATE using Inverse Propensity Weighting."""
    X = dmatrix(ps_formula, df)
    t = df[T].values
    y = df[Y].values

    ps_model = LogisticRegression(penalty=None, max_iter=1000).fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]

    ate = np.mean((t - ps) / (ps * (1 - ps)) * y)
    return float(ate)


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """Estimate the ATE using the Doubly Robust estimator."""
    X = np.asarray(dmatrix(formula, df))
    t = df[T].values
    y = df[Y].values

    ps_model = LogisticRegression(penalty=None, max_iter=1000).fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]

    mu0 = LinearRegression().fit(X[t == 0], y[t == 0]).predict(X)
    mu1 = LinearRegression().fit(X[t == 1], y[t == 1]).predict(X)

    ate = (
        np.mean(t * (y - mu1) / ps + mu1)
        - np.mean((1 - t) * (y - mu0) / (1 - ps) + mu0)
    )
    return float(ate)
