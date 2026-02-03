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

def build_regime_features(
        df: pd.DataFrame,
        start_date: str = '2005-01-01',
        zscore_window: int = 252
) -> pd.DataFrame:
    '''
    Build stationary regime feature matrix for HMM estimation.

    Features:
    1. move_chg: Change in MOVE index (rates volatility)
    2. vix_chg: Change in VIX (equity volatility)
    3. dxy_chg: Change in DXY (USD stress)
    4. cesi_usd_zscore: Rolling z-score of US economic surprise index
    5. cesi_eur_zscore: Rolling z-score of EUR economic surprise index
    6. policy_fed_ecb_spread: Fed Funds - ECB rate
    7. policy_fed_boj_spread: Fed Funds - BoJ rate

    All features are:
    - I(0) stationary
    - Lagged by 1 day (no lookahead bias)
    - Daily frequency (business days)
     
    :param df: Master Dataframe
    :type df: pd.DataFrame
    :param start_date: Start date for feature extraction (2005-01-01 as DXY only starts from there)
    :type start_date: str
    :param zscore_window: Rolling window for z-score calculation (default to 1 year)
    :type zscore_window: int
    :return: Feature matrix with DatetimeIndex, ready for HMM fitting
    :rtype: DataFrame

    Notes:
    - CESI substitutes for unavailable inflation data
    '''

    df = df.loc[start_date:].copy()

    features = pd.DataFrame(index=df.index)

    # MOVE index changes
    move_col = 'move__MOVE Index'
    if move_col in df.columns:
        features['move_chg'] = df[move_col].diff()
    else:
        raise KeyError(f"MOVE index not found. Expected column: {move_col}")
    
    # VIX changes
    vix_col = 'us_eq__VIX Index'
    if vix_col in df.columns:
        features['vix_chg'] = df[vix_col].diff()
    else:
        raise KeyError(f"VIX not found. Expected column: {vix_col}")
    
    # DXY changes
    dxy_col = 'dxy__DXY Index'
    if dxy_col in df.columns:
        features['dxy_chg'] = df[dxy_col].diff()
    else:
        raise KeyError(f"DXY not found. Expected column: {dxy_col}")
    
    # CESI rolling z-scores
    cesi_usd_col = 'cesi__CESIUSD Index'
    cesi_eur_col = 'cesi__CESIEUR Index'

    if cesi_usd_col in df.columns:
        features['cesi_usd_zscore'] = rolling_zscore(
            df[cesi_usd_col], window=zscore_window
        )
    else:
        raise KeyError(f"CESI USD not found. Expected column: {cesi_usd_col}")
    
    if cesi_eur_col in df.columns:
        features['cesi_eur_zscore'] = rolling_zscore(
            df[cesi_eur_col], window=zscore_window
        )
    else:
        raise KeyError(f"CESI EUR not found. Expected column: {cesi_eur_col}")

    # Policy Rate Divergence
    policy_div = compute_policy_divergence(df)
    features = features.join(policy_div)

    # Lag all features by 1 day to prevent lookahead
    features = features.shift(1)
    
    features = features.dropna()
    features = features.asfreq('B')
    
    return features

def validate_stationarity(features: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    '''
    Docstring for validate_stationarity
    
    :param features: Feature matrix from build_regime_features()
    :type features: pd.DataFrame
    :param alpha: Significance level for stationarity test
    :type alpha: float
    :return: Stationarity test results from the test suite
    :rtype: DataFrame

    Raises:
    AssertionError if any feature is non-stationary
    '''
    from src.diagnostics.stationarity import run_stationarity_suite
    
    results = run_stationarity_suite(features, alpha=alpha)
    
    # Check for any non-stationary features
    non_stationary = results[results['label'].str.contains('I\\(1\\)', case=False, na=False)]
    
    if len(non_stationary) > 0:
        raise AssertionError(
            f"Non-stationary features detected:\n{non_stationary[['column', 'label']]}"
        )
    
    return results