'''
Explicitly document what every column in master_df represents, whether it’s a level or change, and how it’s used in the strategy.
'''

import pandas as pd
from pathlib import Path

MASTER_PATH = Path("DATA/processed/master_df.parquet")
OUT_MD = Path("docs/variable_map.md")

def classify(col: str) -> dict:
    dataset, series = col.split("__", 1)

    category = "unknown"
    level_or_change = "level"
    intended_use = "unknown"
    notes = ""


    if dataset == "bond_yields":
        category = "Core market (rates)"
        level_or_change = "level (yields)"
        intended_use = "Cointegration (levels); changes for correlation/PCA/clustering"

    elif dataset == "policyrates":
        category = "Policy & macro drivers"
        level_or_change = "level"
        intended_use = "Policy divergence; regime conditioning"

    elif dataset == "cesi":
        category = "Policy & macro drivers"
        level_or_change = "level"
        intended_use = "Macro surprise state; regime features"

    elif dataset in ["move", "dxy", "repo", "fx_ov_iv", "us_eq", "fx_1m"]:
        category = "Stress & regime indicators"
        level_or_change = "level + change"
        intended_use = "HMM regime inference; breakdown diagnostics; risk filters"

    if "MOVE" in series:
        notes = "Rates implied volatility proxy"
    elif "BBDXY" in series:
        notes = "Broad USD funding stress proxy"
    elif "VIX" in series:
        notes = "Equity risk regime proxy"
    elif "SPX" in series:
        notes = "Risk-on / risk-off indicator"
    elif "SOFR" in series or "ESTR" in series:
        notes = "Short-term funding stress proxy"

    return {
        "column": col,
        "dataset": dataset,
        "category": category,
        "level_vs_change": level_or_change,
        "intended_use": intended_use,
        "notes": notes,
    }

def main():
    df = pd.read_parquet(MASTER_PATH)

    rows = [classify(c) for c in df.columns]
    table = pd.DataFrame(rows).sort_values(["category", "column"])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Variable Map\n\n")
        f.write(
            "This table documents what each series represents, how it is treated "
            "(levels vs changes), and how it is used in the strategy pipeline.\n\n"
        )
        f.write(table.to_markdown(index=False))

    print("Wrote:", OUT_MD)

if __name__ == "__main__":
    main()
