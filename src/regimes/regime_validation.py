import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from pathlib import Path


"""
Regime vs Structure validation helpers.

This module provides:
- merge_regime_and_structure
- descriptive regime stats
- hypothesis testing helpers
- loaders that convert *pair-level* structural files into *daily* scalars

Task 3.3 expects DAILY structural series aligned by Date index:
- rolling_corr_mean (daily average across pairs)
- cluster_persistence (daily average) OR cluster_fragmentation (daily average)
- pc1_variance (already daily)
"""


def merge_regime_and_structure(df_regimes, df_structure):
    common_idx = df_regimes.index.intersection(df_structure.index)
    return df_regimes.loc[common_idx].join(df_structure.loc[common_idx])


def calculate_regime_stats(df_merged, structure_cols, regime_col="regime_most_likely"):
    return df_merged.groupby(regime_col)[structure_cols].agg(["mean", "std", "count"])


def test_hypothesis(df_merged, metric, stress_regime, normal_regime):
    stress_data = df_merged[df_merged["regime_most_likely"] == stress_regime][metric].dropna()
    normal_data = df_merged[df_merged["regime_most_likely"] == normal_regime][metric].dropna()

    t_stat, p_val = ttest_ind(stress_data, normal_data, equal_var=False)

    return {
        "metric": metric,
        "stress_mean": float(stress_data.mean()) if len(stress_data) else np.nan,
        "normal_mean": float(normal_data.mean()) if len(normal_data) else np.nan,
        "diff": float(stress_data.mean() - normal_data.mean()) if (len(stress_data) and len(normal_data)) else np.nan,
        "p_value": float(p_val) if len(stress_data) and len(normal_data) else np.nan,
        "significant": bool(p_val < 0.05) if len(stress_data) and len(normal_data) else False,
    }


def plot_regime_boxplots(df_merged, metrics):
    plt.figure(figsize=(5 * len(metrics), 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        sns.boxplot(x="regime_most_likely", y=metric, data=df_merged)
        plt.title(f"{metric} by Regime")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------

def _find_date_column(df: pd.DataFrame):
    # try common names first
    for c in ["date", "Date", "datetime", "time", "timestamp"]:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c

    # fallback: scan columns and find one that parses like a date
    for c in df.columns:
        if df[c].dtype == object or "date" in str(c).lower() or "time" in str(c).lower():
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c

    return None


def _pick_value_column(df: pd.DataFrame, prefer_keywords):
    # numeric candidates only
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None

    # prefer columns with keywords
    lower = {c: str(c).lower() for c in num_cols}
    for kw in prefer_keywords:
        for c in num_cols:
            if kw in lower[c]:
                return c

    # fallback: pick the numeric column with largest non-null coverage
    coverage = [(c, df[c].notna().mean()) for c in num_cols]
    coverage.sort(key=lambda x: x[1], reverse=True)
    return coverage[0][0]


def _to_daily_series_from_pair_file(csv_path: Path, value_keywords):
    """
    Converts a pair-level CSV (tenor/currency in index etc.) into a DAILY scalar series:
      - detect date column
      - set date as index
      - pick best numeric value column
      - groupby(date).mean()
    """
    df = pd.read_csv(csv_path)

    date_col = _find_date_column(df)
    if date_col is None:
        return None, f"NO_DATE_COL in {csv_path.name}"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    if df.empty:
        return None, f"DATE_PARSE_FAILED in {csv_path.name}"

    val_col = _pick_value_column(df, value_keywords)
    if val_col is None:
        return None, f"NO_NUMERIC_COL in {csv_path.name}"

    s = df.groupby(date_col)[val_col].mean()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index()
    return s, f"OK ({csv_path.name} | date_col={date_col} | val_col={val_col})"


def load_optional_structure_series(results_dir: Path):
    """
    Returns DAILY series dict if the files exist.

    Keys (if found):
      - rolling_corr_mean : pd.Series indexed by Date
      - cluster_persistence : pd.Series indexed by Date
      - cluster_fragmentation : pd.Series indexed by Date

    Also returns:
      - _rolling_corr_file, _cluster_file for audit trail
      - _rolling_corr_status, _cluster_status
    """
    results_dir = Path(results_dir)

    out = {
        "rolling_corr_mean": None,
        "cluster_persistence": None,
        "cluster_fragmentation": None,
        "_rolling_corr_file": None,
        "_cluster_file": None,
        "_rolling_corr_status": None,
        "_cluster_status": None,
    }

    # Rolling corr (pair-level -> daily mean)
    rolling_candidates = [
        results_dir / "rolling_corr_pair_metrics.csv",
        results_dir / "rolling_corr_metrics.csv",
        results_dir / "rolling_corr.csv",
    ]
    rolling_file = next((p for p in rolling_candidates if p.exists()), None)
    if rolling_file is not None:
        s, status = _to_daily_series_from_pair_file(
            rolling_file,
            value_keywords=["rolling", "corr", "correlation", "rho"]
        )
        out["_rolling_corr_file"] = str(rolling_file)
        out["_rolling_corr_status"] = status
        out["rolling_corr_mean"] = s

    # Clustering (pair-level -> daily mean)
    cluster_candidates = [
        results_dir / "cluster_pair_persistence.csv",
        results_dir / "cluster_persistence.csv",
        results_dir / "cluster_fragmentation.csv",
        results_dir / "n_clusters.csv",
    ]
    cluster_file = next((p for p in cluster_candidates if p.exists()), None)
    if cluster_file is not None:
        # if it looks like fragmentation/n_clusters -> treat as fragmentation
        name = cluster_file.name.lower()
        if "fragment" in name or "n_clusters" in name:
            s, status = _to_daily_series_from_pair_file(
                cluster_file,
                value_keywords=["fragment", "n_cluster", "nclusters", "clusters"]
            )
            out["cluster_fragmentation"] = s
            out["_cluster_status"] = status
        else:
            s, status = _to_daily_series_from_pair_file(
                cluster_file,
                value_keywords=["persist", "stability", "cohesion", "score"]
            )
            out["cluster_persistence"] = s
            out["_cluster_status"] = status

        out["_cluster_file"] = str(cluster_file)

    return out


# -----------------------------
# Hypothesis gate helpers 
# -----------------------------

def pick_stress_and_normal_regimes(stats_summary, pc1_metric="pc1_variance"):
    stress = stats_summary[pc1_metric]["mean"].idxmax()
    normal = stats_summary[pc1_metric]["mean"].idxmin()
    return int(stress), int(normal)


def hypothesis_yesno(df_master, metric, stress_regime, normal_regime, direction="gt"):
    r = test_hypothesis(df_master, metric, stress_regime, normal_regime)

    if direction == "gt":
        ok_dir = (r["diff"] > 0)
    else:
        ok_dir = (r["diff"] < 0)

    ans = "YES" if (r["significant"] and ok_dir) else "NO"

    return {
        "answer": ans,
        "metric": metric,
        "stress_mean": r["stress_mean"],
        "normal_mean": r["normal_mean"],
        "diff": r["diff"],
        "p_value": r["p_value"],
        "direction_tested": direction,
    }


# -----------------------------
# Script runner
# -----------------------------

def run_task33_validation(results_dir: Path):
    results_dir = Path(results_dir)

    df_regimes = pd.read_csv(results_dir / "regime_probabilities.csv", index_col=0, parse_dates=True)
    df_pca = pd.read_csv(results_dir / "pca_5Y_rolling_pc1_metrics.csv", index_col=0, parse_dates=True)

    df_regimes.index = pd.to_datetime(df_regimes.index).tz_localize(None)
    df_pca.index = pd.to_datetime(df_pca.index).tz_localize(None)

    pc1_col = df_pca.columns[0]
    pc1_var = df_pca[[pc1_col]].rename(columns={pc1_col: "pc1_variance"})

    df_master = merge_regime_and_structure(df_regimes[["regime_most_likely"]], pc1_var)

    opt = load_optional_structure_series(results_dir)
    print("Rolling corr status:", opt.get("_rolling_corr_status"))
    print("Cluster status:", opt.get("_cluster_status"))
    print("Rolling corr file:", opt.get("_rolling_corr_file"))
    print("Cluster file:", opt.get("_cluster_file"))

    if opt.get("rolling_corr_mean") is not None:
        df_master = df_master.join(opt["rolling_corr_mean"].rename("rolling_corr_mean"), how="inner")

    if opt.get("cluster_persistence") is not None:
        df_master = df_master.join(opt["cluster_persistence"].rename("cluster_persistence"), how="inner")

    if opt.get("cluster_fragmentation") is not None:
        df_master = df_master.join(opt["cluster_fragmentation"].rename("cluster_fragmentation"), how="inner")

    metrics = ["pc1_variance"]
    for c in ["rolling_corr_mean", "cluster_persistence", "cluster_fragmentation"]:
        if c in df_master.columns:
            metrics.append(c)

    stats_summary = calculate_regime_stats(df_master, metrics)
    stats_summary.to_csv(results_dir / "regime_structure_comparison.csv")

    stress_regime, normal_regime = pick_stress_and_normal_regimes(stats_summary, pc1_metric="pc1_variance")

    rows = []

    # Q1 weaken? (rolling corr drops in stress)
    if "rolling_corr_mean" in df_master.columns:
        r = hypothesis_yesno(df_master, "rolling_corr_mean", stress_regime, normal_regime, direction="lt")
        rows.append({"question": "Do relationships weaken in stress regimes?", **r})
    else:
        rows.append({"question": "Do relationships weaken in stress regimes?", "answer": "NOT AVAILABLE", "reason": "missing rolling_corr_mean"})

    # Q2 pc1 dominates? (pc1 rises in stress)
    r = hypothesis_yesno(df_master, "pc1_variance", stress_regime, normal_regime, direction="gt")
    rows.append({"question": "Does PC1 dominate during stress?", **r})

    # Q3 clusters fragment? (fragmentation rises OR persistence falls)
    if "cluster_fragmentation" in df_master.columns:
        r = hypothesis_yesno(df_master, "cluster_fragmentation", stress_regime, normal_regime, direction="gt")
        rows.append({"question": "Do clusters fragment during stress?", **r})
    elif "cluster_persistence" in df_master.columns:
        r = hypothesis_yesno(df_master, "cluster_persistence", stress_regime, normal_regime, direction="lt")
        rows.append({"question": "Do clusters fragment during stress?", **r})
    else:
        rows.append({"question": "Do clusters fragment during stress?", "answer": "NOT AVAILABLE", "reason": "missing cluster metric"})

    df_answers = pd.DataFrame(rows)
    df_answers.to_csv(results_dir / "regime_hypothesis_answers.csv", index=False)

    return stats_summary, df_answers


if __name__ == "__main__":
    # run from repo root: python src/regimes/regime_validation.py
    results_dir = Path(__file__).resolve().parents[2] / "results"
    stats, answers = run_task33_validation(results_dir)
    print("\nSaved:")
    print(" - results/regime_structure_comparison.csv")
    print(" - results/regime_hypothesis_answers.csv")
    print("\nHypothesis answers:")
    print(answers)
