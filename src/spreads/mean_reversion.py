import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.regimes.utils import get_continuous_regime_periods

def estimate_pooled_mean_reversion(
    y_A: pd.Series,
    y_B: pd.Series,
    alpha_r: float,
    beta_r: float,
    regime_series: pd.Series,
    target_state: int,
    min_obs: int = 60,
    min_spell_obs: int = 10
) -> dict:
    """
    Estimate mean-reversion speed (AR1 alpha) for a regime-specific spread.
    
    Uses regime-specific OLS params (alpha_r, beta_r) to construct the spread
    within each spell, then de-means each spell before pooling.
    This avoids shared-intercept bias from heterogeneous spell levels.
    """
    spells = get_continuous_regime_periods(regime_series, target_state, min_duration=min_spell_obs)

    y_list = []
    x_list = []
    total_valid_obs = 0

    for start, end in spells:
        y_a_block = y_A.loc[start:end]
        y_b_block = y_B.loc[start:end]
        common = y_a_block.dropna().index.intersection(y_b_block.dropna().index)
        if len(common) < min_spell_obs:
            continue

        # Static spread using regime-level cointegrating vector
        spread = y_a_block.loc[common] - alpha_r - beta_r * y_b_block.loc[common]

        # *** SPELL DEMEANING: removes heterogeneous equilibrium levels across spells ***
        spread = spread - spread.mean()

        delta = spread.diff()
        lag   = spread.shift(1)
        mask  = delta.notna() & lag.notna()

        if mask.sum() > 0:
            y_list.append(delta[mask])
            x_list.append(lag[mask])
            total_valid_obs += mask.sum()

    if total_valid_obs < min_obs:
        return {'valid': False, 'reason': f'Insufficient data (n={total_valid_obs})'}

    y_pooled = pd.concat(y_list)
    X_pooled = sm.add_constant(pd.concat(x_list))

    try:
        results = sm.OLS(y_pooled, X_pooled).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        ar1_alpha = results.params.iloc[1]
        p_val     = results.pvalues.iloc[1]
        half_life = -np.log(2) / ar1_alpha if ar1_alpha < 0 else np.inf

        return {
            'valid': True,
            'alpha': ar1_alpha,
            'p_value': p_val,
            'half_life': half_life,
            'n_obs': total_valid_obs
        }
    except Exception as e:
        return {'valid': False, 'reason': str(e)}