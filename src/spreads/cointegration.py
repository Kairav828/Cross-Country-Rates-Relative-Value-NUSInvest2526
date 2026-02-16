""""
Engle-Granger cointegration testing with residual diagnostics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def schwert_lag_length(n_obs: int ) -> int:
    """
    Calculate ADF lag length using Schwert criterion.
    
    :param n_obs: Number of Observations
    :type n_obs: int
    :return: Recommended Lag Length
    :rtype: int
    """

    return int(12 * (n_obs/100) ** 0.25)

def engle_granger_test(
        y_A: pd.Series,
        y_B: pd.Series,
        adf_method: str = 'schwert',
        max_lags: Optional[int] = None,
        alpha: float = 0.05
) -> Dict:
    """
    Two-step Engle-Granger cointegration test.

    Step 1: Estimate OLS regression y_A = alpha + beta*y_B + epsilon
    Step 2: Test Residuals for unit root (ADF test)
    
    :param y_A: Dependent variable (yield series A)
    :type y_A: pd.Series
    :param y_B: Independent variable (yield series B)
    :type y_B: pd.Series
    :param adf_method: Method for lag selection: 'schwert', 'aic', or 'fixed'
    :type adf_method: str
    :param max_lags: Maximum lags for ADF test
    :type max_lags: Optional[int]
    :param alpha: Significance level for cointegration test
    :type alpha: float
    :return: Test results including
                - hedge_ratio_beta: OLS coefficient beta
                - adf_stat: ADF test statistic on residuals
                - adf_pval: p-value for ADF test
                - cointegrated: bool, True if reject null hypothesis
                - n_obs: sample size
                - residuals: regression residuals
                - alpha: intercept
                - r_squared: OLS R^2
    :rtype: Dict
    """
    df = pd.DataFrame({'y_A': y_A, 'y_B': y_B}).dropna()

    if len(df) < 30:
        raise ValueError(f"Insufficient Observations: {len(df)} < 30")
    
    # Step 1: OLS Regression
    X = add_constant(df['y_B']) # adds a column of 1s to y_B, enabling regression model to estimate an intercept alpha (would force line through origin otherwise)
    model = OLS(df['y_A'], X).fit()

    alpha_intercept = model.params['const']
    beta = model.params.iloc[1] 
    residuals = model.resid
    r_squared = model.rsquared

    # Step 2: ADF test on residuals
    if adf_method == 'schwert':
        n_lags = schwert_lag_length(len(df)) if max_lags is None else max_lags
    elif adf_method == 'aic':
        n_lags = None # ADF auto select
    elif adf_method == 'fixed':
        n_lags = max_lags if max_lags is not None else 5
    else:
        raise ValueError(f"Unknown adf_method: {adf_method}")
    
    # ADF Test: Null hypothesis of ADF test is unit root (no cointegration)
    adf_result = adfuller(residuals, maxlag=n_lags, regression='c', autolag='AIC' if adf_method == 'aic' else None)

    adf_stat = adf_result[0]
    adf_pval = adf_result[1]
    adf_lags_used = adf_result[2]

    # Reject null if p_value < alpha -> cointegrated
    cointegrated = adf_pval < alpha

    return {
        'hedge_ratio_beta': beta,
        'alpha': alpha_intercept,
        'adf_stat': adf_stat,
        'adf_pval': adf_pval,
        'adf_lags_used': adf_lags_used,
        'cointegrated': cointegrated,
        'n_obs': len(df),
        'residuals': residuals,
        'r_squared': r_squared
    }

def validate_residuals(
        residuals: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        lags: int = 10
) -> Dict:
    """
    Run residual diagnostics: heteroskedasticity and autocorrelation tests.
    
    :param residuals: Regression residuals
    :type residuals: pd.Series
    :param exog: Exogenous variable for White test (if None, tests vs constant only)
    :type exog: Optional[pd.DataFrame]
    :param lags: Number of lags for Ljung-Box test
    :type lags: int
    :return: Following results:
                - white_stat: White test statistic
                - white_pval: White test p-value
                - ljung_box_stat: Ljung-Box Q statistic
                - ljung_box_pval: Ljung-Box p-value
                - heteroskedastic: bool (True if White p < 0.05)
                - autocorrelated: bool (True if Ljung-Box p < 0.05)
    :rtype: Dict
    """

    results = {}

    # White test for heteroskedasticity
    if exog is not None:
        white_result = het_white(residuals, exog)
        results['white_stat'] = white_result[0]
        results['white_pval'] = white_result[1]
    else:
        # Test against a constant only
        X_const = add_constant(pd.Series(np.ones(len(residuals)), index=residuals.index))
        white_result = het_white(residuals, X_const)
        results['white_stat'] = white_result[0]
        results['white_pval'] = white_result[1]
    
    results['heteroskedastic'] = results['white_pval'] < 0.05

    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=False)
    results['ljung_box_stat'] = lb_result[0][0]
    results['ljung_box_pval'] = lb_result[1][0]
    results['autocorrelated'] = results['ljung_box_pval'] < 0.05

    return results


def regime_conditional_cointegration(
        y_A: pd.Series,
        y_B: pd.Series,
        regime_labels: pd.Series,
        min_obs: int = 100,
        adf_method: str = 'schwert',
        alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test cointegration separately for each regime state.
    
    :param y_A: Yield series with datetime index
    :type y_A: pd.Series
    :param y_B: Yield series with datetime index
    :type y_B: pd.Series
    :param regime_labels: Regime state lables (0, 1, 2, 3) with datetime index
    :type regime_labels: pd.Series
    :param min_obs: Minimum observations required per regime
    :type min_obs: int
    :param adf_method: ADF lag selection method
    :type adf_method: str
    :param alpha: Significance level
    :type alpha: float
    :return: One row per regime state with all test statistics
    :rtype: DataFrame
    """

    df = pd.DataFrame({
        'y_A': y_A,
        'y_B': y_B,
        'regime': regime_labels
    }).dropna()

    results = []

    for regime_state in sorted(df['regime'].unique()):
        mask = df['regime'] == regime_state
        regime_data = df[mask]

        n_obs = len(regime_data)
        sample_warning = n_obs < 300 

        if n_obs < min_obs:
            # Insufficient data - record as null
            results.append({
                'regime_state': int(regime_state),
                'n_obs': n_obs,
                'sample_warning': True,
                'cointegrated': None,
                'adf_stat': np.nan,
                'adf_pval': np.nan,
                'hedge_ratio_beta': np.nan,
                'note': f'Insufficient data ({n_obs} < {min_obs})'
            })
            continue

        # Engle-Granger test
        try:
            eg_result = engle_granger_test(
                regime_data['y_A'],
                regime_data['y_B'],
                adf_method=adf_method,
                alpha=alpha
            )

            X = add_constant(regime_data['y_B'])
            resid_diag = validate_residuals(eg_result['residuals'], exog=X, lags=10)

            row = {
                'regime_state': int(regime_state),
                'n_obs': n_obs,
                'sample_warning': sample_warning,
                'cointegrated': eg_result['cointegrated'],
                'adf_stat': eg_result['adf_stat'],
                'adf_pval': eg_result['adf_pval'],
                'adf_lags_used': eg_result['adf_lags_used'],
                'hedge_ratio_beta': eg_result['hedge_ratio_beta'],
                'alpha': eg_result['alpha'],
                'r_squared': eg_result['r_squared'],
                'white_stat': resid_diag['white_stat'],
                'white_pval': resid_diag['white_pval'],
                'heteroskedastic': resid_diag['heteroskedastic'],
                'ljung_box_stat': resid_diag['ljung_box_stat'],
                'ljung_box_pval': resid_diag['ljung_box_pval'],
                'autocorrelated': resid_diag['autocorrelated'],
                'note': ''
            }
            
            results.append(row)

        except Exception as e:
            results.append({
                'regime_state': int(regime_state),
                'n_obs': n_obs,
                'sample_warning': sample_warning,
                'cointegrated': None,
                'adf_stat': np.nan,
                'adf_pval': np.nan,
                'hedge_ratio_beta': np.nan,
                'note': f'Error: {str(e)}'
            })
    
    return pd.DataFrame(results)