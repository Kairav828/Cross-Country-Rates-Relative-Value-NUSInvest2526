"""
Purpose:
--------
Generate a formal audit of data coverage and missingness for the master dataset.

Why this exists:
----------------
Many statistical procedures used later in the project (rolling PCA, clustering,
HMM regime detection, cointegration tests) are highly sensitive to sample windows.
If variables have different start/end dates or missingness patterns, results can
change silently due to sample drift rather than economic effects.

This script creates:
1) results/missingness_report.csv  — machine-readable audit
2) docs/data_coverage.md           — human-readable summary
"""

from pathlib import Path
import pandas as pd

MASTER_PATH = Path("DATA/processed/master_df.parquet")
OUT_CSV = Path("results/missingness_report.csv")
OUT_MD = Path("docs/data_coverage.md")

def make_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "start_date": df.apply(lambda s: s.first_valid_index()),
        "end_date": df.apply(lambda s: s.last_valid_index()),
        "n_obs": df.count(),
        "n_total": len(df),
        "missing_pct": (df.isna().sum() / len(df) * 100.0).round(2),
    })
    report.index.name = "variable"
    return report.sort_values(["missing_pct", "n_obs"], ascending=[False, True])

def write_md_summary(report: pd.DataFrame, df: pd.DataFrame) -> str:
    start = df.index.min()
    end = df.index.max()
    md = []
    md.append("# Data Coverage & Missingness\n")
    md.append(f"- Date range: {start} -> {end}\n")
    md.append(f"- Rows: {len(df)}\n")
    md.append(f"- Columns: {df.shape[1]}\n")
    md.append("\n## Top Missing Series (by % missing)\n")
    md.append(report.head(15)[["missing_pct","start_date","end_date","n_obs"]].to_markdown())
    md.append("\n")
    return "\n".join(md)

if __name__ == "__main__":
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH}. Run build_master first.")

    df = pd.read_parquet(MASTER_PATH).sort_index()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    report = make_missingness_report(df)
    report.to_csv(OUT_CSV)

    OUT_MD.write_text(write_md_summary(report, df))
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_MD)
