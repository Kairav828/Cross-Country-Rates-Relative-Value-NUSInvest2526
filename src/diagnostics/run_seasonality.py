"""
Purpose:
--------
Run seasonality diagnostics across relevant Δ-series in the master dataset and
produce a consolidated results table.

This script generates:
- results/seasonality_tests.csv

The table is an auditable artifact used to justify:
- 'model_explicitly' vs 'ignore' decisions
- whether month dummies are needed later
"""

from pathlib import Path
import pandas as pd
from src.diagnostics.seasonality import analyze_seasonality, run_task2

MASTER = Path("DATA/processed/master_df.parquet")
OUT = Path("results/seasonality_tests.csv")

FILTERS = {
    "yields": lambda c: "bond_yields" in c.lower() or "us10y" in c.lower() or "us2y" in c.lower(),
    "funding": lambda c: "repo" in c.lower() or "sofr" in c.lower(),
    "stress": lambda c: "move" in c.lower() or "dxy" in c.lower(),
}

def main():
    if not MASTER.exists():
        raise FileNotFoundError("Missing DATA/processed/master_df.parquet. Run build_master first.")

    # Load levels
    df = pd.read_parquet(MASTER).sort_index()

    # Existing pipeline uses Δseries
    df_diff = df.diff()

    rows = []
    selected_cols = set()

    # -----------------------------
    # Existing auditable table (seasonality_tests.csv)
    # -----------------------------
    for grp, pred in FILTERS.items():
        cols = [c for c in df_diff.columns if pred(c)]
        selected_cols.update(cols)

        for col in cols:
            res = analyze_seasonality(df_diff[col], name=col)
            rows.append({
                "group": grp,
                "variable": col,
                "ye_ratio": res["ye_ratio"],
                "ye_p": res["ye_p"],
                "month_var_p": res["month_var_p"],
                "month_dummy_p": res["month_dummy_p"],
                "decision": res["decision"],
            })

    out = pd.DataFrame(rows).sort_values(["decision", "month_dummy_p", "month_var_p", "ye_p"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Saved:", OUT)

    # -----------------------------
    # summary artifacts (seasonality_summary.csv + seasonality_note.md)
    # Uses the SAME selected columns for reproducibility.
    # run_task2() internally diffs the series, so we pass LEVEL series from df[col].
    # -----------------------------
    selected_cols = sorted(selected_cols)
    if len(selected_cols) > 0:
        run_task2(df, selected_cols, output_dir="results")
        print("Saved: results/seasonality_summary.csv")
        print("Saved: results/seasonality_note.md")
    else:
        print("No seasonality columns selected by FILTERS; skipping run_task2().")


if __name__ == "__main__":
    main()
