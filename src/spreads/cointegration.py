"""
Engle-Granger cointegration testing with residual diagnostics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from src.regimes.utils import get_continuous_regime_periods


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
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    results['ljung_box_stat'] = lb_result['lb_stat'].iloc[0]
    results['ljung_box_pval'] = lb_result['lb_pvalue'].iloc[0]
    results['autocorrelated'] = results['ljung_box_pval'] < 0.05

    return results


def spell_adf_fisher(
        y_A: pd.Series,
        y_B: pd.Series,
        alpha_r: float,
        beta_r: float,
        spells: list,
        min_spell_obs: int = 40,
        adf_method: str = 'schwert'
) -> dict:
    """
    Run ADF on each contiguous regime spell independently, then combine p-values using Fisher's method. Avoids ADF across discontinuous observations.

    H0: all spells have a unit root (no cointegration in any spell.)
    Reject -> cointegration holds within-spell.
    
    :param y_A: Yield series with datetime index
    :type y_A: pd.Series
    :param y_B: Yield series with datetime index
    :type y_B: pd.Series
    :param alpha_r: Alpha from OLS
    :type alpha_r: float
    :param beta_r: Beta from OLS
    :type beta_r: float
    :param spells: list of periods of contiguous regimes
    :type spells: list
    :param min_spell_obs: minimum observations required in a contiguous regime
    :type min_spell_obs: int
    :param adf_method: method to determine # lags in adf
    :type adf_method: str
    :return: Dictionary with regime-conditional cointegration test results, including:
        - ``valid`` (bool): whether a combined Fisher test was computed (at least one spell met ``min_spell_obs``).
        - ``reason`` (str): explanation when ``valid`` is ``False``; empty or informational otherwise.
        - ``fisher_stat`` (float): Fisher combined test statistic from spell-level ADF p-values.
        - ``fisher_pval`` (float): p-value associated with ``fisher_stat``.
        - ``n_spells_used`` (int): number of spells contributing to the Fisher combination.
        - ``spell_results`` (list[dict]): per-spell diagnostics with keys ``start``, ``end``, ``n_obs``, ``adf_stat``, ``adf_pval``.
    :rtype: dict
    """

    p_values = []
    spell_results = []

    for start, end in spells:
        y_a_spell = y_A.loc[start:end].dropna()
        y_b_spell = y_B.loc[start:end].dropna()

        common = y_a_spell.index.intersection(y_b_spell.index)
        if len(common) < min_spell_obs:
            continue

        # Residuals from the regime-level cointegrating vector
        resid = y_a_spell.loc[common] - alpha_r - beta_r * y_b_spell.loc[common]

        n_lags = schwert_lag_length(len(resid)) if adf_method == 'schwert' else None
        adf_result = adfuller(resid, maxlag=n_lags, regression='c', autolag='AIC' if adf_method == 'aic' else None)

        p_values.append(adf_result[1])
        spell_results.append({
            'start': start, 'end': end,
            'n_obs': len(common),
            'adf_stat': adf_result[0],
            'adf_pval': adf_result[1]
        })
    if len(p_values) == 0:
        return {
            'valid': False,
            'reason': f'No spells met min_spell_obs={min_spell_obs}',
            'spell_results': spell_results
        }

    if len(p_values) == 1:
        # Single spell — return its ADF p-value directly
        # Fisher on 1 test is just -2*ln(p) ~ chi2(2), equivalent to the raw p
        return {
            'valid': True,
            'fisher_chi2': -2 * np.log(p_values[0]),
            'fisher_pval': p_values[0],   # single spell p-value used directly
            'n_spells_used': 1,
            'spell_results': spell_results,
            'note': 'Single spell — Fisher not combined, raw ADF p-value used'
        }
    # if len(p_values) < 2:
    #     return {
    #         'valid': False,
    #         'reason': f'Only {len(p_values)} spell(s) met min_spell_obs={min_spell_obs}',
    #         'spell_results': spell_results
    #     }
    
    chi2_stat = -2 * np.sum(np.log(p_values))
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))

    return {
        'valid': True,
        'fisher_chi2': chi2_stat,
        'fisher_pval': combined_p,
        'n_spells_used': len(p_values),
        'spell_results': spell_results
    }

def regime_conditional_cointegration(
        y_A: pd.Series,
        y_B: pd.Series,
        regime_labels: pd.Series,
        min_obs: int = 100,
        min_spell_obs: int = 40,
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
    :param min_spell_obs: Minimum observations required per valid spell
    :type min_spell_obs: int
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
            # Step 1: OLS on all regime observations — valid because OLS is i.i.d.
            X = add_constant(regime_data['y_B'])
            model = OLS(regime_data['y_A'], X).fit()
            alpha_r = model.params['const']
            beta_r  = model.params.iloc[1]
            r_sq    = model.rsquared

            # Step 2: Per-spell ADF + Fisher combination
            spells = get_continuous_regime_periods(
                regime_labels, int(regime_state), min_duration=min_spell_obs
            )
            fisher_res = spell_adf_fisher(
                y_A, y_B, alpha_r, beta_r, spells, min_spell_obs, adf_method=adf_method
            )

            if not fisher_res['valid']:
                results.append({
                    'regime_state': int(regime_state), 'n_obs': n_obs,
                    'cointegrated': None, 'note': fisher_res['reason'],
                    'hedge_ratio_beta': beta_r, 'alpha': alpha_r
                })
                continue

            cointegrated = fisher_res['fisher_pval'] < alpha

            row = {
                'regime_state':    int(regime_state),
                'n_obs':           n_obs,
                'sample_warning':  sample_warning,
                'cointegrated':    cointegrated,
                'fisher_chi2':     fisher_res['fisher_chi2'],
                'fisher_pval':     fisher_res['fisher_pval'],
                'n_spells_used':   fisher_res['n_spells_used'],
                'hedge_ratio_beta': beta_r,
                'alpha':           alpha_r,
                'r_squared':       r_sq,
                'note':            ''
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