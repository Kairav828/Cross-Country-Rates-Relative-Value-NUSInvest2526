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



Week 2 seasonlity decision logic implemented:
1) Identify cadidate series with seasonality from week 1 (repo, fx vol, move, dxy, yield changes)
2) For each series, run:
   - Month dummy regression on Δseries
   - Joint F-test for seasonality
   - Dec–Jan vs Feb–Nov variance ratio
3) Classify seasonality strength (NONE / WEAK / STRONG) based on:
- STRONG seasonality: F-test p < 0.05 and YE vol ratio > 1.3
    → "Add month dummies or treat Dec–Jan as separate regime"
- WEAK seasonality: YE vol ratio > 1.2
    → "Ignore in model but treat Dec–Jan as risk overlay"
- NONE: otherwise
    → "Ignore seasonality"

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
    - Joint F-test for seasonality
    - Dec–Jan vs Feb–Nov variance ratio

    Returns a dict of summary statistics.
    """
    series = series.dropna()
    # FIXED: Bloomberg object dtype - Force numeric
    series = pd.to_numeric(series, errors="coerce").dropna()
    # Compute daily changes
    dseries = series.diff().dropna()

    # Month dummies
    months = dseries.index.month
    month_dummies = pd.get_dummies(months, prefix="m", drop_first=True)

    # Convert dummies to float
    month_dummies = month_dummies.astype(float)

    # Add intercept
    X = sm.add_constant(month_dummies)

    # Ensure regression inputs are numeric arrays
    y = dseries.astype(float).values
    X = X.astype(float).values

    # Run OLS safely
    model = sm.OLS(y, X).fit()

    f_pval = model.f_pvalue

    # Variance split: Dec–Jan vs Feb–Nov
    dec_jan_mask = dseries.index.month.isin([12, 1])
    rest_mask = dseries.index.month.isin([2,3,4,5,6,7,8,9,10,11])

    var_decjan = np.var(dseries[dec_jan_mask])
    var_rest = np.var(dseries[rest_mask])

    vol_ratio = var_decjan / var_rest if var_rest > 0 else np.nan

    return {
        "YE Vol Ratio (Dec/Jan vs Rest)": vol_ratio,
        "F-Test p-value": f_pval
    }



def classify_seasonality(vol_ratio: float, pval: float):
    """
    Assigns NONE / WEAK / STRONG seasonality label
    based on statistical + economic thresholds.
    """

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
        pval = stats["F-Test p-value"]

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
