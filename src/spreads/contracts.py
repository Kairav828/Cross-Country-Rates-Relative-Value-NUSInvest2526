from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class PairContract:
    pair_id: str
    y_col: str
    x_col: str
    tenor: str
    ccy_leg_1: str
    ccy_leg_2: str
    tradable_regimes: List[int]
    cointegration_source: str
    status: str
    notes: str = ""


def _parse_regime_list(value) -> List[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        out = ast.literal_eval(value)
        return [int(x) for x in out]
    return []


def load_pair_contracts(csv_path: str | Path = "results/contracts/approved_pairs.csv") -> List[PairContract]:
    df = pd.read_csv(csv_path)
    contracts = []

    for _, row in df.iterrows():
        contracts.append(
            PairContract(
                pair_id=row["pair_id"],
                y_col=row["y_col"],
                x_col=row["x_col"],
                tenor=row["tenor"],
                ccy_leg_1=row["ccy_leg_1"],
                ccy_leg_2=row["ccy_leg_2"],
                tradable_regimes=_parse_regime_list(row["tradable_regimes"]),
                cointegration_source=row["cointegration_source"],
                status=row["status"],
                notes=row.get("notes", ""),
            )
        )

    return contracts


def load_active_contracts(csv_path: str | Path = "results/contracts/approved_pairs.csv") -> List[PairContract]:
    contracts = load_pair_contracts(csv_path)
    return [c for c in contracts if c.status.lower() == "approved"]


def get_contract_by_pair_id(pair_id: str, csv_path: str | Path = "results/contracts/approved_pairs.csv") -> PairContract:
    contracts = load_pair_contracts(csv_path)
    for c in contracts:
        if c.pair_id == pair_id:
            return c
    raise ValueError(f"Pair contract not found: {pair_id}")