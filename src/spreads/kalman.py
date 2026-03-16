from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_dynamic_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
    init_alpha: float = 0.0,
    init_beta: float = 1.0,
) -> pd.DataFrame:
    """
    Random-walk state-space model:

        y_t = alpha_t + beta_t * x_t + eps_t
        state_t = state_{t-1} + eta_t

    state = [alpha_t, beta_t]

    Returns DataFrame with:
        y, x, alpha_t, beta_t, spread_t, innovation_t, innovation_var_t, innovation_z_t
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna().copy()
    n = len(df)

    if n < 30:
        raise ValueError(f"Not enough observations for Kalman filter: {n}")

    theta = np.zeros((2, n))         # [alpha_t, beta_t]
    P = np.zeros((2, 2, n))          # covariance of state

    theta[:, 0] = np.array([init_alpha, init_beta], dtype=float)
    P[:, :, 0] = np.eye(2)

    # state noise covariance
    W = (delta / (1.0 - delta)) * np.eye(2)

    # observation noise variance
    V = obs_var

    innovation = np.zeros(n)
    innovation_var = np.zeros(n)

    for t in range(1, n):
        theta_pred = theta[:, t - 1]
        P_pred = P[:, :, t - 1] + W

        F_t = np.array([1.0, df["x"].iloc[t]], dtype=float)
        y_t = float(df["y"].iloc[t])

        y_hat = F_t @ theta_pred
        Q_t = float(F_t @ P_pred @ F_t.T + V)
        K_t = (P_pred @ F_t) / Q_t

        innovation[t] = y_t - y_hat
        innovation_var[t] = Q_t

        theta[:, t] = theta_pred + K_t * innovation[t]
        P[:, :, t] = P_pred - np.outer(K_t, F_t) @ P_pred

    out = df.copy()
    out["alpha_t"] = theta[0, :]
    out["beta_t"] = theta[1, :]
    out["spread_t"] = out["y"] - out["alpha_t"] - out["beta_t"] * out["x"]
    out["innovation_t"] = innovation
    out["innovation_var_t"] = innovation_var
    out["innovation_z_t"] = np.where(
        innovation_var > 0,
        innovation / np.sqrt(innovation_var),
        np.nan,
    )

    return out


def run_kalman_for_contract(
    yields_df: pd.DataFrame,
    pair_id: str,
    y_col: str,
    x_col: str,
    out_dir: str = "results/spreads",
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> pd.DataFrame:
    out = kalman_dynamic_hedge_ratio(
        y=yields_df[y_col],
        x=yields_df[x_col],
        delta=delta,
        obs_var=obs_var,
    )

    out["pair_id"] = pair_id

    PathLike = __import__("pathlib").Path
    PathLike(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(f"{out_dir}/{pair_id}_kalman.parquet")

    return out