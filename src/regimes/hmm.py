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
    Compute average feature values per regime state for economic interpretation.
    Uses relative ranking across states instead of fixed thresholds.
    '''
    
    state_summary = []
    
    for state in range(model.n_components):
        mask = labels['regime_most_likely'] == state
        state_data = features[mask]
        means = state_data.mean()
        
        summary = {'state': state}
        summary.update(means.to_dict())
        summary['n_days'] = mask.sum()
        summary['pct_days'] = 100 * mask.sum() / len(features)
        
        state_summary.append(summary)
    
    df_summary = pd.DataFrame(state_summary)
    
    # Compute relative rankings for each feature across states
    # Z-score: (state_mean - mean_across_states) / std_across_states
    feature_cols = ['move_chg', 'dxy_chg', 'curve_signal', 'macro_surprise']
    
    for col in feature_cols:
        cross_state_mean = df_summary[col].mean()
        cross_state_std = df_summary[col].std()
        df_summary[f'{col}_zscore'] = (df_summary[col] - cross_state_mean) / cross_state_std
    
    # Now assign labels based on relative z-scores
    labels_list = []
    
    for idx, row in df_summary.iterrows():
        move_z = row['move_chg_zscore']
        dxy_z = row['dxy_chg_zscore']
        macro_z = row['macro_surprise_zscore']
        curve_z = row['curve_signal_zscore']
        
        # Define "high" as >0.5 std above other states, "low" as <-0.5 std
        THRESHOLD_Z = 0.5
        
        # PRIORITY 1: VOLATILITY
        if move_z > THRESHOLD_Z:
            # This state has distinctly higher MOVE than other states
            if macro_z < -THRESHOLD_Z:
                label = 'Crisis / Growth Collapse'
            elif dxy_z > THRESHOLD_Z:
                label = 'Crisis / USD Funding Stress'
            elif curve_z > THRESHOLD_Z:
                label = 'Crisis / Flight-to-Quality'
            else:
                label = 'High Volatility / Uncertainty'
        
        elif move_z < -THRESHOLD_Z:
            # This state has distinctly lower MOVE than other states
            if macro_z > THRESHOLD_Z and curve_z > THRESHOLD_Z:
                label = 'Stable / Reflationary Growth'
            elif macro_z > THRESHOLD_Z:
                label = 'Stable / Positive Surprises'
            elif curve_z < -THRESHOLD_Z:
                label = 'Stable / Late Cycle'
            else:
                label = 'Low Volatility / Risk-On'
        
        # PRIORITY 2: USD STRESS
        elif dxy_z > THRESHOLD_Z:
            if macro_z < -THRESHOLD_Z:
                label = 'USD Strength / Growth Divergence'
            else:
                label = 'USD Strength / Safe Haven'
        
        elif dxy_z < -THRESHOLD_Z:
            label = 'USD Weakness / Convergence'
        
        # PRIORITY 3: MACRO/CURVE
        elif macro_z > THRESHOLD_Z and curve_z > THRESHOLD_Z:
            label = 'Growth Acceleration / Reflation'
        elif macro_z < -THRESHOLD_Z and curve_z < -THRESHOLD_Z:
            label = 'Slowdown / Flattening'
        elif macro_z > THRESHOLD_Z:
            label = 'Positive Surprises'
        elif macro_z < -THRESHOLD_Z:
            label = 'Negative Surprises'
        elif curve_z > THRESHOLD_Z:
            label = 'Steepening / Easing Expectations'
        elif curve_z < -THRESHOLD_Z:
            label = 'Flattening / Tightening'
        else:
            label = 'Mixed / Transitional'
        
        labels_list.append(label)
    
    df_summary['economic_label'] = labels_list
    
    return df_summary


def compute_regime_persistence(labels: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute average duration of each regime state.
    
    :param labels: Output from extract_regime_labels() with 'regime_most_likely' column
    :type labels: pd.DataFrame
    :return: Columns: state, avg_duration_days, min_duration_days, max_duration_days
    :rtype: DataFrame
    '''
    
    regime_seq = labels['regime_most_likely'].values
    
    # Find regime switches
    switches = np.where(np.diff(regime_seq) != 0)[0] + 1
    switches = np.concatenate([[0], switches, [len(regime_seq)]])
    
    # Compute durations for each spell
    durations = []
    
    for i in range(len(switches) - 1):
        start = switches[i]
        end = switches[i + 1]
        duration = end - start
        state = regime_seq[start]
        durations.append({'state': state, 'duration': duration})
    
    durations_df = pd.DataFrame(durations)
    
    # Aggregate by state
    persistence = durations_df.groupby('state')['duration'].agg([
        ('avg_duration_days', 'mean'),
        ('min_duration_days', 'min'),
        ('max_duration_days', 'max'),
        ('n_spells', 'count')
    ]).reset_index()
    
    return persistence