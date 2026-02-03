'''
Regime Feature Engineering for HMM Estimation.

Constructs I(0) macro and volatility features that represent regime state:
- Rates volatility: MOVE index changes
- Equity volatility: VIX changes
- USD stress: DXY changes
- Growth momentum: CESI rolling z-scores
- Policy divergence: short-end rate spreads

All features are stationary (I(0)) and lagged 1 day to prevent lookahead bias.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

def rolling_zscore(s: pd.Series, window: int=252, min_periods: int=126) -> pd.Series:
    '''
    Rolling z-score normalization.
    
    :param s: Input series
    :type s: pd.Series
    :param window: Rolling window in days (default 252 = ~1 year)
    :type window: int
    :param min_periods: Minimum observations required (default 126 = ~6 months)
    :type min_periods: int
    :return: Z-scored series
    :rtype: Series[Any]
    '''

    roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
    roll_std = s.rolling(window=window, min_periods=min_periods).std()

    roll_std = roll_std.replace(0, np.nan)

    return (s - roll_mean) / roll_std

def compute_policy_divergence(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute policy rate divergence metrics.

    Extracts Fed - ECB and Fed - BoJ spreads as indicators of monetary policy fragmentation.
    
    :param df: Master dataframe with columns: policyrates__FDTR Index, policyrates__EURR002W Index, policyrates__BOJDTR Index
    :type df: pd.DataFrame
    :return: Dataframe with columns: policy_fed_ecb_spread, policy_fed_boj_spread
    :rtype: DataFrame
    '''

    out = pd.DataFrame(index=df.index)

    # Fed Funds to ECB Deposit Rate
    fed = df.get('policyrates__FDTR Index')
    ecb = df.get('policyrates__EURR002W Index')

    if fed is not None and ecb is not None:
        out['policy_fed_ecb_spread'] = fed - ecb

    # Fed Funds to BoJ Rate
    boj = df.get('policyrates__BOJDTR Index')

    if fed is not None and boj is not None:
        out['policy_fed_boj_spread'] = fed - boj

    return out