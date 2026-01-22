'''
Function to output data to DATA/processed
'''

from pathlib import Path
from src.data.load_raw import load_all_raw

OUT_PATH = Path("DATA/processed/master_df.parquet")

def build_master_df():
    dfs = load_all_raw()

    master = None
    for name, df in dfs.items():
        df = df.copy()
        df.columns = [f"{name}__{c}" for c in df.columns]

        master = df if master is None else master.join(df, how="outer")

    master = master.sort_index()
    return master

if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_master_df()
    df.to_parquet(OUT_PATH)

    print("Saved:", OUT_PATH)
    print("Rows:", len(df), "Cols:", df.shape[1])
    print("Date range:", df.index.min(), "->", df.index.max())
