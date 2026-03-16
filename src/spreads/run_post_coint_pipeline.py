from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.spreads.contracts import get_contract_by_pair_id
from src.spreads.kalman import run_kalman_for_contract
from src.spreads.ou import estimate_ou_by_regime, attach_reversion_probability
from src.spreads.thresholds import add_threshold_features, add_signal_flags
from src.spreads.signals import apply_regime_gate, build_trade_signals


PAIR_ID = "USD_EUR_5Y"


def load_daily_regime_series(regime_probs_path: str) -> pd.Series:
    """
    Load daily most-likely regime labels from regime_probabilities.csv
    Expected columns:
      - date
      - regime_most_likely
    """
    regime_df = pd.read_csv(regime_probs_path, parse_dates=["date"])
    regime_df = regime_df.set_index("date").sort_index()

    if "regime_most_likely" not in regime_df.columns:
        raise ValueError(
            "regime_probabilities.csv must contain a column named 'regime_most_likely'"
        )

    regime_series = regime_df["regime_most_likely"].astype(int)
    regime_series.name = "regime"
    return regime_series


def main():
    Path("results/spreads").mkdir(parents=True, exist_ok=True)
    Path("results/signals").mkdir(parents=True, exist_ok=True)
    Path("results/ou").mkdir(parents=True, exist_ok=True)

    # ACTUAL PATHS FROM YOUR REPO
    yields_path = "DATA/processed/master_df.parquet"
    regimes_path = "results/regime_probabilities.csv"

    contract = get_contract_by_pair_id(PAIR_ID)

    # Load data
    yields_df = pd.read_parquet(yields_path)
    regime_series = load_daily_regime_series(regimes_path)

    # Defensive checks
    missing_cols = [c for c in [contract.y_col, contract.x_col] if c not in yields_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {yields_path}: {missing_cols}\n"
            f"Available sample columns: {list(yields_df.columns)[:20]}"
        )

    # --------------------------------------------------
    # 2. Kalman dynamic spread
    # --------------------------------------------------
    spread_df = run_kalman_for_contract(
        yields_df=yields_df,
        pair_id=contract.pair_id,
        y_col=contract.y_col,
        x_col=contract.x_col,
        out_dir="results/spreads",
        delta=1e-4,
        obs_var=1e-3,
    )

    # join regime labels on common dates
    common_idx = spread_df.index.intersection(regime_series.index)
    spread_df = spread_df.loc[common_idx].copy()
    spread_df["regime"] = regime_series.loc[common_idx]

    spread_df.to_parquet(f"results/spreads/{contract.pair_id}_kalman_with_regime.parquet")

    # --------------------------------------------------
    # 3. OU by regime on Kalman spread
    # --------------------------------------------------
    ou_params_df = estimate_ou_by_regime(
        spread_df=spread_df,
        regime_series=spread_df["regime"],
        pair_id=contract.pair_id,
        min_obs=30,
    )
    ou_params_df.to_csv(f"results/ou/{contract.pair_id}_ou_params.csv", index=False)

    # --------------------------------------------------
    # 4. threshold features + OU probability
    # --------------------------------------------------
    signal_df = add_threshold_features(
        spread_df=spread_df,
        z_window=60,
    )

    signal_df = attach_reversion_probability(
        signal_df=signal_df,
        ou_params_df=ou_params_df[ou_params_df["valid"] == True].copy(),
        horizon_days=20,
        band_width=0.25,
    )

    signal_df = add_signal_flags(
        signal_df,
        entry_z=2.0,
        exit_z=0.5,
        min_prob_revert=0.55,
        require_prob_filter=True,
    )

    # --------------------------------------------------
    # 5. modular regime gating from pair contract
    # --------------------------------------------------
    signal_df = apply_regime_gate(
        signal_df,
        tradable_regimes=contract.tradable_regimes,
    )

    signal_df = build_trade_signals(signal_df)

    signal_df["pair_id"] = contract.pair_id
    signal_df.to_parquet(f"results/signals/{contract.pair_id}_signals.parquet")

    print("\nDone.")
    print(f"Pair: {contract.pair_id}")
    print(f"Y column: {contract.y_col}")
    print(f"X column: {contract.x_col}")
    print(f"Tradable regimes from contract: {contract.tradable_regimes}")
    print("\nLast 10 rows of signals:")
    print(signal_df.tail(10)[[
        "pair_id",
        "regime",
        "regime_allowed",
        "spread_t",
        "spread_z",
        "prob_revert_horizon",
        "long_signal",
        "short_signal",
        "exit_signal",
    ]])


if __name__ == "__main__":
    main()