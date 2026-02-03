'''
Hidden Markov Model Regime Estimation.

Fits Gaussian HMM to macro/vol features to identify latent regime states.
Include model selection via BIC and state interpretation tools.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Tuple


def fit_hmm_models(
        features: pd.DataFrame,
        n_states_range: list[int] = [2,3,4],
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
) -> dict[int, GaussianHMM]:
    '''
    Fit Gaussian HMM for multiple state counts.
     
    :param features: Stationary Feature Matrix (from build_regime_features)
    :type features: pd.DataFrame
    :param n_states_range: Number of states to try (default[2,3,4])
    :type n_states_range: list[int]
    :param covariance_type: HMM covariance structure ('full' (each component has its own general covariance matrix), 'tied', 'diag', 'spherical')
    :type covariance_type: str
    :param n_iter: Maximum EM iterations
    :type n_iter: int
    :param random_state: Random seed
    :type random_state: int
    :return: Mapping of n_states -> fitted model
    :rtype: dict[int, Any]
    '''

    # Convert to numpy array
    X = features.values

    models = {}

    for n_states in n_states_range:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )

        model.fit(X)
        models[n_states] = model

    return models

def compute_bic(model: GaussianHMM, features: pd.DataFrame) -> float:
    '''
    Bayesian Information Criterion for fitted HMM

    BIC = -2 * log_likelihood + k * log(n)
    where k = number of free parameters, n = number of observations

    Lower BIC indicates better model
    
    :param model: Fitted HMM
    :type model: GaussianHMM
    :param features: Feature Matrix used for fitting 
    :type features: pd.DataFrame
    :return: BIC value 
    :rtype: float
    '''

    X = features.values
    log_likelihood = model.score(X)

    n_samples, n_features = X.shape
    n_states = model.n_components

    # Count free parameters
    # Transition matrix: n_states * (n_states - 1)
    # Means: n_states * n_features
    # Covariances: depends on covariance_type
    if model.covariance_type == 'full':
        # Full covariance: n_states * n_features * (n_features + 1) / 2
        cov_params = n_states * n_features * (n_features + 1) // 2
    elif model.covariance_type == 'diag':
        cov_params = n_states * n_features
    elif model.covariance_type == 'spherical':
        cov_params = n_states
    elif model.covariance_type == 'tied':
        cov_params = n_features * (n_features + 1) // 2
    else:
        cov_params = 0
    
    n_params = n_states * (n_states - 1) + n_states * n_features + cov_params
    
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return bic


def select_best_model(
        models: dict[int, GaussianHMM],
        features: pd.DataFrame
) -> Tuple[GaussianHMM, int, pd.DataFrame]:
    '''
    Select best HMM via BIC comparison
    
    :param models: Fitted models from fit_hmm_models()
    :type models: dict[int, GaussianHMM]
    :param features: Feature matrix
    :type features: pd.DataFrame
    :return: Tuple of [Model with lowest BIC, Optimal number of states, BIC comparison table]
    :rtype: Tuple[Any, int, DataFrame]
    '''

    bic_results = []
    
    for n_states, model in models.items():
        bic = compute_bic(model, features)
        log_lik = model.score(features.values)
        
        bic_results.append({
            'n_states': n_states,
            'log_likelihood': log_lik,
            'bic': bic
        })
    
    comparison_df = pd.DataFrame(bic_results).sort_values('bic')
    
    best_row = comparison_df.iloc[0]
    best_n_states = int(best_row['n_states'])
    best_model = models[best_n_states]
    
    return best_model, best_n_states, comparison_df


def extract_regime_labels(
        model: GaussianHMM,
        features: pd.DataFrame
) -> pd.DataFrame:
    '''
    Extract regime probabilities and most likely path from fitted HMM
    
    :param model: Fitted HMM
    :type model: GaussianHMM
    :param features: Feature matrix with DatetimeIndex
    :type features: pd.DataFrame
    :return: Dataframe with columns: regime_prob_0, regime_prob_1, ..., regime_most_likely and index with dates
    :rtype: DataFrame
    '''

    X = features.values
    
    # Filtered state probabilities (forward algorithm)
    state_probs = model.predict_proba(X)
    
    # Most likely state sequence (Viterbi algorithm)
    most_likely = model.predict(X)
    
    # Build output dataframe
    out = pd.DataFrame(
        state_probs,
        index=features.index,
        columns=[f'regime_prob_{i}' for i in range(model.n_components)]
    )

    out['regime_most_likely'] = most_likely
    
    return out


def interpret_states(
        model: GaussianHMM,
        features: pd.DataFrame,
        labels: pd.DataFrame
) -> pd.DataFrame:
    '''
    Compute average feature values per regime state for economic interpretation
    
    :param model: Fitted HMM
    :type model: GaussianHMM
    :param features: Feature matrix
    :type features: pd.DataFrame
    :param labels: Output from extract_regime_labels() with 'regime_most_likely' column
    :type labels: pd.DataFrame
    :return: Rows = states, columns = mean feature values + economic_label
    :rtype: DataFrame
    '''

    state_summary = []
    
    for state in range(model.n_components):
        # Get dates assigned to this state
        mask = labels['regime_most_likely'] == state
        state_data = features[mask]
        
        # Compute mean of each feature in this state
        means = state_data.mean()
        
        summary = {'state': state}
        summary.update(means.to_dict())
        
        # Add derived metrics
        summary['n_days'] = mask.sum()
        summary['pct_days'] = 100 * mask.sum() / len(features)
        
        # Assign economic label based on feature signature
        # High MOVE/VIX = stress, High DXY = USD strength, Negative CESI = growth slowdown
        if means.get('move_chg', 0) > 0.5 and means.get('vix_chg', 0) > 0.5:
            economic_label = 'Stress / High Volatility'
        elif means.get('move_chg', 0) < -0.5 and means.get('vix_chg', 0) < -0.5:
            economic_label = 'Stable / Low Volatility'
        elif means.get('dxy_chg', 0) > 0.2:
            economic_label = 'USD Strength / Funding Stress'
        elif means.get('cesi_usd_zscore', 0) < -0.3:
            economic_label = 'Growth Slowdown'
        else:
            economic_label = 'Mixed / Transitional'
        
        summary['economic_label'] = economic_label
        
        state_summary.append(summary)
    
    return pd.DataFrame(state_summary)