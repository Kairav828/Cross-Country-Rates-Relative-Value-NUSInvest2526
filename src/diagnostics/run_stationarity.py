import pandas as pd
from pathlib import Path
from src.diagnostics.stationarity import run_stationarity_suite

MASTER = Path("DATA/processed/master_df.parquet")
OUT = Path("results/stationarity_tests.csv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MASTER)

    # Levels
    res_levels = run_stationarity_suite(df)
    res_levels["transform"] = "level"

    # First differences
    ddf = df.diff()
    res_diff = run_stationarity_suite(ddf)
    res_diff["transform"] = "diff"

    res = pd.concat([res_levels, res_diff], ignore_index=True)
    res.to_csv(OUT, index=False)

    print("Saved:", OUT)
    print(res.groupby(["transform", "label"]).size())


if __name__ == "__main__":
    main()
