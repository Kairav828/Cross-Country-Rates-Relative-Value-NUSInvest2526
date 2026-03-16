from __future__ import annotations

import pandas as pd


def apply_regime_gate(
    df: pd.DataFrame,
    tradable_regimes: list[int],
) -> pd.DataFrame:
    out = df.copy()
    out["regime_allowed"] = out["regime"].isin(tradable_regimes)
    return out


def build_trade_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final modular signal layer.

    A setup only becomes a trade signal if the regime gate allows it.
    """
    out = df.copy()

    out["long_signal"] = out["regime_allowed"] & out["long_setup"]
    out["short_signal"] = out["regime_allowed"] & out["short_setup"]
    out["exit_signal"] = out["exit_setup"]

    return out