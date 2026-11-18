## Assignment 2

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def _ate_and_se(data: pd.DataFrame) -> Tuple[float, float]:
## Return the difference-in-means ATE estimate and its standard error.

    y_treat = data.loc[data["T"] == 1, "Y"]
    y_ctrl = data.loc[data["T"] == 0, "Y"]

    n1 = len(y_treat)
    n0 = len(y_ctrl)

    # ATE = E[Y|T=1] - E[Y|T=0]
    ate = y_treat.mean() - y_ctrl.mean()

    # SE(ATE) = sqrt(Var(Y|T=1)/n1 + Var(Y|T=0)/n0), using sample variances
    var1 = y_treat.var(ddof=1)
    var0 = y_ctrl.var(ddof=1)
    se = np.sqrt(var1 / n1 + var0 / n0)

    return float(ate), float(se)


def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
## Confidence interval for the ATE.

    ate, se = _ate_and_se(data)

    # z_(alpha/2): 1.96 for alpha = 0.05
    z = norm.ppf(1 - alpha / 2)

    ci_lower = ate - z * se
    ci_upper = ate + z * se

    return float(ate), float(ci_lower), float(ci_upper)


def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
## Two-sided test of H0

    ate, se = _ate_and_se(data)

    # t = (ATE_estimate - 0) / SE(ATE)
    t_stat = (ate - 0) / se

    # p-value = 2 * (1 - Phi(|t|))
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return float(ate), float(t_stat), float(p_value)
