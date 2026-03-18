import pandas as pd

def build_spread_position(row: pd.Series) -> int:
    """
    Translates boolean signal flags into a target position.
    +1 -> Long spread
    -1 -> Short spread
     0 -> Flat
    """
    if row["long_signal"]:
        return 1
    
    if row["short_signal"]:
        return -1
    
    return 0

def add_positions_to_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the build_spread_position logic to the entire dataframe."""
    df = df.copy()
    df["target_position"] = df.apply(build_spread_position, axis=1)
    return df