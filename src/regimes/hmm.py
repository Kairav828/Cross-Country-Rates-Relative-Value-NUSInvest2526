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