"""
Rolling Cointegration tests within regimes.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from .cointegration import engle_granger_test

def rolling_cointegration(
        y_A: pd.Series,
        y_B: pd.Series,
        window: int = 252,
        min_periods: int = 100,
        adf_method: str = 'schwert',
        alpha: float = 0.05
) -> pd.DataFrame:
    """
    Rolling Engle-Granger cointegration test
    
    :param y_A: Yield series with datetime index
    :type y_A: pd.Series
    :param y_B: Yield series with datetime index
    :type y_B: pd.Series
    :param window: Rolling window size (default 252 = ~1 trading year)
    :type window: int
    :param min_periods: Minimum observations required
    :type min_periods: int
    :param adf_method: ADF lag selection method
    :type adf_method: str
    :param alpha: Significance level
    :type alpha: float
    :return: Columns
                - window_end_date
                - adf_stat
                - adf_pval
                - hedge_ratio_beta
                - cointegrated
                - n_obs
                - r_squared
    :rtype: DataFrame
    """
    df = pd.DataFrame({'y_A': y_A, 'y_B': y_B}).dropna()

    if len(df) < window:
        raise ValueError(f"Insufficient data: {len(df)} < window size {window}")
    
    results = []

    for i in range(window, len(df) + 1):
        window_data = df.iloc[i - window:i]
        window_end = df.index[i-1]

        if len(window_data) < min_periods:
            continue

        try:
            eg_result = engle_granger_test(
                window_data['y_A'],
                window_data['y_B'],
                adf_method=adf_method,
                alpha=alpha
            )

            results.append({
                'window_end_date': window_end,
                'adf_stat': eg_result['adf_stat'],
                'adf_pval': eg_result['adf_pval'],
                'hedge_ratio_beta': eg_result['hedge_ratio_beta'],
                'cointegrated': eg_result['cointegrated'],
                'n_obs': eg_result['n_obs'],
                'r_squared': eg_result['r_squared']
            })
            
        except Exception as e:
            # Skip windows with errors (e.g., perfect collinearity)
            continue
    
    return pd.DataFrame(results)

def rolling_by_regime(
        y_A: pd.Series,
        y_B: pd.Series,
        regime_labels: pd.Series,
        window: int = 252,
        min_periods: int = 100,
        adf_method: str = 'schwert',
        alpha: float = 0.05
) -> Dict[int, pd.DataFrame]:
    """
    Rolling cointegration tests separately within each regime.
    
    :param y_A: Yield series
    :type y_A: pd.Series
    :param y_B: Yield series
    :type y_B: pd.Series
    :param regime_labels: Regime state labels
    :type regime_labels: pd.Series
    :param window: Rolling window size
    :type window: int
    :param min_periods: Min. Observations per window
    :type min_periods: int
    :param adf_method: ADF lag selection method
    :type adf_method: str
    :param alpha: Significance level
    :type alpha: float
    :return: Keys are regime states, Values are DataFrames with rolling test results
    :rtype: Dict[int, DataFrame]
    """
    df = pd.DataFrame({
        'y_A': y_A,
        'y_B': y_B,
        'regime': regime_labels
    }).dropna()

    results_by_regime = {}

    for regime_state in sorted(df['regime'].unique()):
        # Filter to regime-specific observations only
        regime_mask = df['regime'] == regime_state
        regime_df = df[regime_mask]
        
        if len(regime_df) < window:
            # Not enough data for rolling window
            results_by_regime[int(regime_state)] = pd.DataFrame()
            continue
        
        # Run rolling test on regime-filtered data
        rolling_results = rolling_cointegration(
            regime_df['y_A'],
            regime_df['y_B'],
            window=window,
            min_periods=min_periods,
            adf_method=adf_method,
            alpha=alpha
        )
        
        # Add regime column
        rolling_results['regime_state'] = int(regime_state)
        
        results_by_regime[int(regime_state)] = rolling_results
    
    return results_by_regime

def summarize_rolling_results(rolling_results: pd.DataFrame) -> Dict:
    """
    Summary statistics from rolling cointegration tests
    
    :param rolling_results: Output from rolling_cointegration()
    :type rolling_results: pd.DataFrame
    :return: 
        - pct_windows_cointegrated: % of windows rejecting null
        - mean_adf_stat: Mean ADF statistic
        - mean_hedge_ratio: Mean hedge ratio β
        - std_hedge_ratio: Std dev of β (stability measure)
    :rtype: Dict
    """
    if len(rolling_results) == 0:
        return {
            'n_windows': 0,
            'pct_windows_cointegrated': np.nan,
            'mean_adf_stat': np.nan,
            'mean_hedge_ratio': np.nan,
            'std_hedge_ratio': np.nan
        }
    
    return {
        'n_windows': len(rolling_results),
        'pct_windows_cointegrated': rolling_results['cointegrated'].mean() * 100,
        'mean_adf_stat': rolling_results['adf_stat'].mean(),
        'mean_hedge_ratio': rolling_results['hedge_ratio_beta'].mean(),
        'std_hedge_ratio': rolling_results['hedge_ratio_beta'].std()
    }