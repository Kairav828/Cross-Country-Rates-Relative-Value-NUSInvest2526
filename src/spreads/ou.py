from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def estimate_ou_ar1(spread: pd.Series, hac_lags: int = 5) -> dict:
    """
    Fit:
        dS_t = a + b * S_{t-1} + e_t

    Mean reversion requires b < 0.

    Mapping:
        kappa = -b
        mu = -a / b
        sigma = std(residuals)
        half_life = ln(2) / kappa
    """
    spread = spread.dropna()

    if len(spread) < 30:
        return {"valid": False, "reason": f"Insufficient observations: {len(spread)}"}

    dS = spread.diff()
    lagS = spread.shift(1)

    reg_df = pd.DataFrame({"dS": dS, "lagS": lagS}).dropna()

    X = sm.add_constant(reg_df["lagS"])
    model = sm.OLS(reg_df["dS"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    a = float(model.params["const"])
    b = float(model.params["lagS"])
    pval_b = float(model.pvalues["lagS"])
    resid_sigma = float(np.std(model.resid, ddof=1))

    if b >= 0:
        return {
            "valid": True,
            "mean_reverting": False,
            "a": a,
            "b": b,
            "pval_b": pval_b,
            "kappa": 0.0,
            "mu": np.nan,
            "sigma": resid_sigma,
            "half_life": np.inf,
            "n_obs": int(len(reg_df)),
        }

    kappa = -b
    mu = -a / b
    half_life = np.log(2) / kappa

    return {
        "valid": True,
        "mean_reverting": True,
        "a": a,
        "b": b,
        "pval_b": pval_b,
        "kappa": kappa,
        "mu": mu,
        "sigma": resid_sigma,
        "half_life": half_life,
        "n_obs": int(len(reg_df)),
    }


def estimate_ou_by_regime(
    spread_df: pd.DataFrame,
    regime_series: pd.Series,
    pair_id: str,
    min_obs: int = 30,
) -> pd.DataFrame:
    """
    Estimate OU parameters on the Kalman spread, separately for each regime.
    """
    common_idx = spread_df.index.intersection(regime_series.dropna().index)
    df = spread_df.loc[common_idx].copy()
    df["regime"] = regime_series.loc[common_idx].astype(int)

    rows = []

    for regime_state in sorted(df["regime"].unique()):
        regime_spread = df.loc[df["regime"] == regime_state, "spread_t"].dropna()

        if len(regime_spread) < min_obs:
            rows.append({
                "pair_id": pair_id,
                "regime": int(regime_state),
                "valid": False,
                "reason": f"Insufficient observations: {len(regime_spread)}",
            })
            continue

        res = estimate_ou_ar1(regime_spread)

        row = {
            "pair_id": pair_id,
            "regime": int(regime_state),
            **res,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def ou_terminal_reversion_probability(
    current_spread: float,
    mu: float,
    kappa: float,
    sigma: float,
    horizon_days: int = 20,
    band_width: float = 0.25,
) -> float:
    """
    Approximate probability that S_T lands inside [mu-band_width, mu+band_width].

    This is NOT exact first-passage probability.
    It is a practical terminal-distribution proxy for ranking signal quality.
    """
    if kappa <= 0 or sigma <= 0:
        return np.nan

    mean_T = mu + (current_spread - mu) * np.exp(-kappa * horizon_days)
    var_T = (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * horizon_days))
    std_T = np.sqrt(max(var_T, 1e-12))

    lower_z = (mu - band_width - mean_T) / std_T
    upper_z = (mu + band_width - mean_T) / std_T

    return float(norm.cdf(upper_z) - norm.cdf(lower_z))


def attach_reversion_probability(
    signal_df: pd.DataFrame,
    ou_params_df: pd.DataFrame,
    horizon_days: int = 20,
    band_width: float = 0.25,
) -> pd.DataFrame:
    df = signal_df.copy()

    mu_map = ou_params_df.set_index("regime")["mu"].to_dict()
    kappa_map = ou_params_df.set_index("regime")["kappa"].to_dict()
    sigma_map = ou_params_df.set_index("regime")["sigma"].to_dict()

    probs = []

    for _, row in df.iterrows():
        regime = row["regime"]
        spread = row["spread_t"]

        mu = mu_map.get(regime, np.nan)
        kappa = kappa_map.get(regime, np.nan)
        sigma = sigma_map.get(regime, np.nan)

        if pd.isna(mu) or pd.isna(kappa) or pd.isna(sigma):
            probs.append(np.nan)
            continue

        probs.append(
            ou_terminal_reversion_probability(
                current_spread=float(spread),
                mu=float(mu),
                kappa=float(kappa),
                sigma=float(sigma),
                horizon_days=horizon_days,
                band_width=band_width,
            )
        )

    df["prob_revert_horizon"] = probs
    return df