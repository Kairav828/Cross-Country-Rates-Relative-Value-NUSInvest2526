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
