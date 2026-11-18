# causal-toolkit-ambersophia

A Python package for causal inference: average treatment effect (ATE) estimation for randomized experiments, propensity score methods for observational data, and meta-learners for conditional average treatment effects (CATE).

## Installation

```bash
git clone https://github.com/ambersophia/causal-toolkit-ambersophia.git
cd causal-toolkit-ambersophia

# Install in editable mode
uv pip install -e .

# Run the tests
uv run pytest
```

Requires Python 3.9+. Dependencies (installed automatically): pandas, numpy, scipy, scikit-learn, lightgbm, patsy.

## Quick start

```python
import numpy as np
import pandas as pd
from causal_toolkit_ambersophia import (
    calculate_ate_ci, calculate_ate_pvalue,
    ipw, doubly_robust,
    s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate,
)

# Simulated data with a true treatment effect of 2
rng = np.random.default_rng(0)
n = 1000
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
T = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2))))
Y = 2 * T + x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
df = pd.DataFrame({"x1": x1, "x2": x2, "T": T, "Y": Y})

# 1. Randomized experiment: difference in means
ate, ci_lower, ci_upper = calculate_ate_ci(df)
ate, t_stat, p_value = calculate_ate_pvalue(df)

# 2. Observational data: propensity score methods
ate_ipw = ipw(df, "x1 + x2", T="T", Y="Y")
ate_dr = doubly_robust(df, "x1 + x2", T="T", Y="Y")

# 3. Heterogeneous effects: meta-learners
train, test = df.iloc[:800], df.iloc[800:]
result = x_learner_discrete(train, test, X=["x1", "x2"], T="T", y="Y")
print(result["cate"].mean())
```

## API reference

### `rct` — randomized controlled trials

The data must have a binary treatment column `T` (0/1) and an outcome column `Y`.

| Function | Returns | Description |
|---|---|---|
| `calculate_ate_ci(data, alpha=0.05)` | `(ate, ci_lower, ci_upper)` | Difference-in-means ATE with a normal-approximation (1 − alpha) confidence interval. |
| `calculate_ate_pvalue(data)` | `(ate, t_stat, p_value)` | Two-sided test of H0: ATE = 0. |

### `propensity` — propensity score methods

| Function | Returns | Description |
|---|---|---|
| `ipw(df, ps_formula, T, Y)` | `float` | ATE by inverse propensity weighting. `ps_formula` is a patsy formula for the covariates (e.g. `"x1 + x2"`); propensity scores come from an unpenalized logistic regression. |
| `doubly_robust(df, formula, T, Y)` | `float` | Doubly robust (AIPW) ATE, combining a logistic propensity model with linear outcome models fit separately for treated and control. |

### `meta_learners` — CATE estimation

All four functions share the signature `(train, test, X, T, y)`. `X` is a list of covariate column names, and `T` and `y` are column names. Each returns a copy of `test` with an added `cate` column. Outcome models use LightGBM.

| Function | Treatment | Description |
|---|---|---|
| `s_learner_discrete` | binary | One model μ(X, T); CATE = μ(x, 1) − μ(x, 0). |
| `t_learner_discrete` | binary | Separate outcome models for treated and control; CATE = μ₁(x) − μ₀(x). |
| `x_learner_discrete` | binary | Imputed effects from cross-group predictions, blended by the propensity score: e(x)·τ₀(x) + (1 − e(x))·τ₁(x). |
| `double_ml_cate` | continuous or binary | Double ML (R-learner): 5-fold cross-fitted residuals of T and Y on X, then a weighted model of the residual ratio. |

## Running the tests

```bash
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
