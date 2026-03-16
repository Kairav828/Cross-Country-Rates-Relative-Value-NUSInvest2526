from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    return (series - rolling_mean) / rolling_std


def add_threshold_features(
    spread_df: pd.DataFrame,
    z_window: int = 60,
) -> pd.DataFrame:
    df = spread_df.copy()

    df["spread_z"] = rolling_zscore(df["spread_t"], window=z_window)
    df["abs_spread_z"] = df["spread_z"].abs()

    # Optional: also keep innovation z if Kalman produced it
    if "innovation_z_t" in df.columns:
        df["abs_innovation_z"] = df["innovation_z_t"].abs()
    else:
        df["abs_innovation_z"] = np.nan

    return df


def add_signal_flags(
    df: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    min_prob_revert: float = 0.55,
    require_prob_filter: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    prob_ok = (
        out["prob_revert_horizon"] >= min_prob_revert
        if require_prob_filter and "prob_revert_horizon" in out.columns
        else pd.Series(True, index=out.index)
    )

    out["long_setup"] = (out["spread_z"] <= -entry_z) & prob_ok
    out["short_setup"] = (out["spread_z"] >= entry_z) & prob_ok
    out["exit_setup"] = out["abs_spread_z"] <= exit_z

    return out