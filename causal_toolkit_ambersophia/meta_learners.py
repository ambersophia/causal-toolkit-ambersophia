"""
Week 4 - CATE Estimation: Meta-learners and Double ML

Assisting sources: Claude Opus 5.5 (Anthropic) was used to help write this code.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """S-Learner: one model mu(X, T); CATE(x) = mu(x, 1) - mu(x, 0)."""
    model = LGBMRegressor(verbose=-1)
    model.fit(train[X + [T]], train[y])

    out = test.copy()
    x1 = test[X + [T]].copy()
    x1[T] = 1
    x0 = test[X + [T]].copy()
    x0[T] = 0
    out["cate"] = model.predict(x1) - model.predict(x0)
    return out


def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """T-Learner: separate outcome models for treated and control."""
    treated = train[train[T] == 1]
    control = train[train[T] == 0]

    mu0 = LGBMRegressor(verbose=-1).fit(control[X], control[y])
    mu1 = LGBMRegressor(verbose=-1).fit(treated[X], treated[y])

    out = test.copy()
    out["cate"] = mu1.predict(test[X]) - mu0.predict(test[X])
    return out


def x_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """X-Learner: imputed treatment effects combined with propensity weights."""
    treated = train[train[T] == 1]
    control = train[train[T] == 0]

    # Stage 1: outcome models (as in the T-Learner)
    mu0 = LGBMRegressor(verbose=-1).fit(control[X], control[y])
    mu1 = LGBMRegressor(verbose=-1).fit(treated[X], treated[y])

    # Stage 2: imputed (pseudo) treatment effects
    tau0_target = mu1.predict(control[X]) - control[y]  # control units
    tau1_target = treated[y] - mu0.predict(treated[X])  # treated units

    tau0 = LGBMRegressor(verbose=-1).fit(control[X], tau0_target)
    tau1 = LGBMRegressor(verbose=-1).fit(treated[X], tau1_target)

    # Propensity score
    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(train[X], train[T])
    e = ps_model.predict_proba(test[X])[:, 1]

    out = test.copy()
    out["cate"] = e * tau0.predict(test[X]) + (1 - e) * tau1.predict(test[X])
    return out


def double_ml_cate(train, test, X, T, y) -> pd.DataFrame:
    """Double ML (R-learner style) for continuous treatment."""
    # Partial out X from T and Y with cross-fitting
    t_hat = cross_val_predict(LGBMRegressor(verbose=-1), train[X], train[T], cv=5)
    y_hat = cross_val_predict(LGBMRegressor(verbose=-1), train[X], train[y], cv=5)

    t_res = train[T] - t_hat
    y_res = train[y] - y_hat

    # Transformed outcome and weights
    y_star = y_res / t_res
    w = t_res ** 2

    cate_model = LGBMRegressor(verbose=-1)
    cate_model.fit(train[X], y_star, sample_weight=w)

    out = test.copy()
    out["cate"] = cate_model.predict(test[X])
    return out
