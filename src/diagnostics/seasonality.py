"""
Seasonality diagnostics for cross-country rates RV (Week 2: Seasonality & Data Hygiene)

What we are testing (and why)
-----------------------------
We are NOT trying to "predict yields" with calendar effects.
We are trying to detect whether the calendar introduces systematic distortions that
can break relative-value assumptions or bias residual-based signals.

Two distinct seasonality channels exist:
1) Mean-shift seasonality:
   - Do average daily changes differ by month?
   - Detected via month dummy regression on Δseries.

2) Volatility / stress seasonality:
   - Does year-end (Dec–Jan) have materially higher variance than the rest of the year?
   - Captures funding stress / balance sheet constraints / repo effects.

Why we use month dummy regression (Δseries)
-------------------------------------------
We regress Δseries on month indicators to test if the expected change differs by month:
    Δx_t = α + Σ β_m * D_{m,t} + ε_t
A joint test of {β_m = 0 for all months} asks:
    "Does the calendar matter at all for the mean of Δx?"

Why classical OLS F-tests are unsafe here
-----------------------------------------
Classical OLS F-tests assume homoskedastic errors (constant variance).
Rates and funding-linked series are commonly heteroskedastic:
- Volatility clusters (crisis vs calm)
- Stress regimes (repo/funding pressure, year-end constraints)
- Fat tails / leverage points

If we ignore heteroskedasticity, p-values can be too optimistic and
we can falsely conclude "seasonality exists" when it is really just
time-varying volatility.

Fix: HC3 robust covariance + Wald (joint) test
----------------------------------------------
We keep OLS coefficients but compute robust standard errors (HC3).
Then we run a Wald joint test on all month dummy coefficients:
    H0: β_Feb = β_Mar = ... = β_Dec = 0
This gives a p-value that remains valid under heteroskedasticity.

Outputs and how they are used
-----------------------------
We output an auditable summary used later in pair selection / regime logic:
- Robust joint test p-value for mean seasonality
- Dec–Jan vs Rest variance ratio for stress/vol seasonality
- A classification label:
    STRONG: robust p < 0.05 AND YE vol ratio > 1.3
    WEAK:   YE vol ratio > 1.2
    NONE:   otherwise

Interpretation:
- STRONG → add month dummies OR treat Dec–Jan as a separate regime.
- WEAK   → ignore in model but apply a Dec–Jan risk overlay (position sizing / gating).
- NONE   → ignore seasonality.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

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


def run_seasonality_test(series: pd.Series):
    """
    Runs:
    - Month dummy regression on Δseries
    - Robust joint Wald test for seasonality (HC3)
    - Dec–Jan vs Feb–Nov variance ratio

    Returns dict of summary stats.
    """
    series = series.dropna()
    series = pd.to_numeric(series, errors="coerce").dropna()

    # Ensure datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    dseries = series.diff().dropna()

    months = dseries.index.month
    month_dummies = pd.get_dummies(months, prefix="m", drop_first=True).astype(float)

    X = sm.add_constant(month_dummies).astype(float)
    y = dseries.astype(float).values

    # OLS with robust covariance
    model = sm.OLS(y, X.values).fit(cov_type="HC3")

    # Joint test: all month dummy coefficients = 0
    # X columns = [const, m_2, m_3, ..., m_12] (because drop_first=True)
    k = X.shape[1]
    R = np.zeros((k - 1, k))
    for i in range(1, k):
        R[i - 1, i] = 1.0

    wald_res = model.f_test(R)
    pval = float(wald_res.pvalue)

    dec_jan_mask = dseries.index.month.isin([12, 1])
    rest_mask = dseries.index.month.isin([2,3,4,5,6,7,8,9,10,11])

    var_decjan = np.var(dseries[dec_jan_mask])
    var_rest = np.var(dseries[rest_mask])

    vol_ratio = var_decjan / var_rest if var_rest > 0 else np.nan

    return {
        "YE Vol Ratio (Dec/Jan vs Rest)": vol_ratio,
        "Robust joint test p-value": pval
    }




def classify_seasonality(vol_ratio: float, pval: float):

    if pval < 0.05 and vol_ratio > 1.3:
        return "STRONG", "Add month dummies or treat Dec–Jan as separate regime"

    elif vol_ratio > 1.2:
        return "WEAK", "Ignore in model but treat Dec–Jan as risk overlay"

    else:
        return "NONE", "Ignore seasonality"



def run_task2(df: pd.DataFrame, series_list, output_dir="results"):
    """
    Runs Task 2 pipeline for all candidate series.

    Outputs:
    - seasonality_summary.csv
    - seasonality_note.md
    """

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for name in series_list:

        stats = run_seasonality_test(df[name])

        vol_ratio = stats["YE Vol Ratio (Dec/Jan vs Rest)"]
        pval = stats["Robust joint test p-value"]

        verdict, decision = classify_seasonality(vol_ratio, pval)

        results.append({
            "Series": name,
            "YE Vol Ratio (Dec/Jan vs Rest)": round(vol_ratio, 3),
            "F-Test p-value": round(pval, 4),
            "Verdict": verdict,
            "Handling Rule": decision
        })

    # Save CSV summary
    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "seasonality_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    # Save Markdown note
    md_lines = []
    md_lines.append("# Seasonality Detection Note\n")
    md_lines.append(
        "We tested year-end seasonality using month-dummy regressions "
        "and Dec–Jan variance comparisons.\n"
    )

    for _, row in summary_df.iterrows():
        md_lines.append(f"## {row['Series']}")
        md_lines.append(f"- YE variance ratio: {row['YE Vol Ratio (Dec/Jan vs Rest)']}")
        md_lines.append(f"- F-test p-value: {row['F-Test p-value']}")
        md_lines.append(f"- Verdict: {row['Verdict']}")
        md_lines.append(f"- Decision: {row['Handling Rule']}\n")

    md_path = os.path.join(output_dir, "seasonality_note.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return summary_df
