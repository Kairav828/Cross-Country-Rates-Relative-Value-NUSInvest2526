'''
Robust test for stationarity using ADF and KPSS
ADF null: unit root (non-stationary), if p < alpha, reject null: stationary
KPSS null: stationary, if p < alpha, reject null: non-stationary

Combining both tests:
- If ADF rejects null and KPSS does not reject null: I(0) (stationary)
- If ADF does not reject null and KPSS rejects null: I(1)-like (non-stationary)
- If both reject or both do not reject: ambiguous classification
'''

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _prep_series(s: pd.Series, min_obs: int = 252) -> pd.Series | None:
    s = s.dropna()
    if len(s) < min_obs:
        return None
    if np.isclose(s.std(), 0):
        return None
    return s


def adf_test(s: pd.Series, regression: str = "c") -> dict:
    """
    ADF null: unit root (non-stationary).
    """
    stat, p, usedlag, nobs, _, _ = adfuller(
        s, autolag="AIC", regression=regression
    )
    return {
        "adf_stat": stat,
        "adf_p": p,
        "adf_usedlag": usedlag,
        "adf_nobs": nobs,
    }


def kpss_test(s: pd.Series, regression: str = "c") -> dict:
    """
    KPSS null: stationary.
    """
    stat, p, lags, _ = kpss(
        s, regression=regression, nlags="auto"
    )
    return {
        "kpss_stat": stat,
        "kpss_p": p,
        "kpss_lags": lags,
    }


def classify_stationarity(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha

    if adf_stationary and kpss_stationary:
        return "I(0) (stationary)"
    if (not adf_stationary) and (not kpss_stationary):
        return "I(1)-like (non-stationary)"
    if adf_stationary and (not kpss_stationary):
        return "Trend-stationary / ambiguous"
    if (not adf_stationary) and kpss_stationary:
        return "Near-unit-root / ambiguous"
    return "Unclear"


def run_stationarity_suite(
    df: pd.DataFrame,
    alpha: float = 0.05,
    adf_reg: str = "c",
    kpss_reg: str = "c",
    min_obs: int = 252,
) -> pd.DataFrame:

    rows = []

    for col in df.columns:
        s = _prep_series(df[col], min_obs=min_obs)

        if s is None:
            rows.append({
                "column": col,
                "nobs": int(df[col].dropna().shape[0]),
                "adf_p": np.nan,
                "kpss_p": np.nan,
                "label": "Insufficient data / constant"
            })
            continue

        adf = adf_test(s, regression=adf_reg)
        kps = kpss_test(s, regression=kpss_reg)

        label = classify_stationarity(
            adf["adf_p"], kps["kpss_p"], alpha
        )

        rows.append({
            "column": col,
            "nobs": len(s),
            "adf_p": adf["adf_p"],
            "kpss_p": kps["kpss_p"],
            "label": label,
        })

    return pd.DataFrame(rows).sort_values(["label", "column"])
