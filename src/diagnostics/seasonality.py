"""
Purpose:
--------
Provide formal statistical diagnostics for detecting seasonality in rate changes
and stress indicators.

Why this exists:
----------------
Seasonality (e.g. year-end funding stress, balance-sheet constraints, repo effects)
can distort volatility estimates and invalidate mean-reversion assumptions if not
explicitly tested.

Week 1 seasonality checks implemented:
1) Year-end (Dec–Jan) vs rest-of-year variance test (Levene)
2) Variance comparison across calendar months (Levene across monthly groups)
3) Month-dummy regression on volatility proxy |Δx| (joint F-test)

Outputs are used to make an explicit modeling decision:
- 'model_explicitly'  → seasonality must be handled or conditioned on
- 'ignore'            → no material seasonality detected
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def _safe(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s[s.index.notnull()]

def year_end_test(series: pd.Series):
    s = _safe(series)
    df = s.to_frame("x")
    df["month"] = df.index.month
    ye = df[df["month"].isin([12, 1])]["x"]
    rest = df[~df["month"].isin([12, 1])]["x"]

    if len(ye) < 2 or len(rest) < 2:
        return {"ye_ratio": np.nan, "ye_p": 1.0, "ye_sig": False}

    _, p = stats.levene(ye, rest)
    ratio = ye.std() / rest.std() if rest.std() != 0 else np.nan
    return {"ye_ratio": ratio, "ye_p": p, "ye_sig": p < 0.05}

def month_variance_test(series: pd.Series):
    """
    Tests whether variance differs across the 12 months (Levene across monthly groups).
    """
    s = _safe(series)
    df = s.to_frame("x")
    df["month"] = df.index.month

    groups = [df[df["month"] == m]["x"].values for m in range(1, 13)]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) < 3:
        return {"month_var_p": 1.0, "month_var_sig": False}

    _, p = stats.levene(*groups)
    return {"month_var_p": p, "month_var_sig": p < 0.05}

def month_dummies_test(series: pd.Series, alpha=0.05):
    """
    Regress volatility proxy |Δx| on month dummies and test joint significance (F-test).
    """
    s = _safe(series)
    df = s.to_frame("x")
    df["month"] = df.index.month

    y = df["x"].abs()
    X = pd.get_dummies(df["month"], drop_first=True).astype(float)
    X = sm.add_constant(X)

    if len(y) < X.shape[1] + 5:
        return {"month_dummy_p": 1.0, "month_dummy_sig": False}

    m = sm.OLS(y.values, X.values).fit()

    k = X.shape[1]
    R = np.zeros((k - 1, k))
    for i in range(k - 1):
        R[i, i + 1] = 1.0

    p = float(m.f_test(R).pvalue)
    return {"month_dummy_p": p, "month_dummy_sig": p < alpha}

def analyze_seasonality(series: pd.Series, name="Asset"):
    s = _safe(series)
    df = s.to_frame("x")
    df["month"] = df.index.month
    monthly_vol = df.groupby("month")["x"].std() * np.sqrt(252)

    ye = year_end_test(s)
    mv = month_variance_test(s)
    md = month_dummies_test(s)

    flag = ye["ye_sig"] or mv["month_var_sig"] or md["month_dummy_sig"]
    decision = "model_explicitly" if flag else "ignore"

    return {
        "name": name,
        "ye_ratio": ye["ye_ratio"],
        "ye_p": ye["ye_p"],
        "month_var_p": mv["month_var_p"],
        "month_dummy_p": md["month_dummy_p"],
        "decision": decision,
        "monthly_vol": monthly_vol,
    }

def plot_seasonality_heatmap(df_changes: pd.DataFrame, contains: str, title: str):
    cols = [c for c in df_changes.columns if contains in c.lower()]
    if not cols:
        print(f"No columns found for filter: {contains}")
        return

    col = cols[0]
    s = df_changes[col].dropna()
    if s.empty:
        print(f"Skipping {title}: no data")
        return

    tmp = s.to_frame(col)
    tmp["year"] = tmp.index.year
    tmp["month"] = tmp.index.month
    pivot = tmp.pivot_table(index="year", columns="month", values=col, aggfunc="std")

    if pivot.empty or pivot.isnull().all().all():
        print(f"Skipping {title}: insufficient data")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm", annot=False, cbar_kws={"label": "Monthly Volatility"})
    plt.title(f"Seasonality Heatmap: {title} ({col})")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()
