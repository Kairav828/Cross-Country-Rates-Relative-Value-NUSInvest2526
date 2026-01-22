'''
Function to load and use raw data
'''

import pandas as pd
from pathlib import Path

RAW_DIR = Path("DATA/raw")

def _read_bbg_csv(path: Path) -> pd.DataFrame:
    """
    Bloomberg-style CSV:
      row 0: field
      row 1: date
      row 2+: data
    """
    df = pd.read_csv(path)
    df = df.iloc[2:].copy()

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_all_raw() -> dict[str, pd.DataFrame]:
    files = {
        "bond_yields": "bond_yields.csv",
        "policyrates": "policyrates.csv",
        "cesi": "citi_economic_surprise_index.csv",
        "move": "OCE BofA MOVE INDEX.csv",
        "dxy": "BBDXY.csv",
        "repo": "repo.csv",
        "us_eq": "us_equity_indicies.csv",
        "fx_1m": "currency_1M_outright_normalizedtousdbaseccy.csv",
        "fx_ov_iv": "currency_Overnight_ATM_Implied_Vol.csv",
    }

    out = {}
    for k, fname in files.items():
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        out[k] = _read_bbg_csv(path)

    return out
