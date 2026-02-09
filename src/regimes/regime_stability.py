import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment  # <--- NEW IMPORT

'''
Functions to assess the stability of HMM regime assignments
under perturbations like random seed changes and noise injection
Metrics: 
- Agreement Percentage
- Adjusted Rand Index (ARI)

'''

def perturb_data(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def fit_perturbed_hmm(X, n_states, random_state=None):
    # Using 'full' covariance as standard
    model = GaussianHMM(n_components=n_states, covariance_type="full", 
                        n_iter=100, random_state=random_state, tol=1e-4)
    model.fit(X)
    return model.predict(X)

def calculate_stability_metrics(baseline_states, alternative_states):
    """
    Compares baseline regimes vs. an alternative run using Optimal Matching.
    """
    # 1. Compute Confusion Matrix
    # Rows = Baseline, Cols = Alternative
    cm = confusion_matrix(baseline_states, alternative_states)
    
    # 2. Find Optimal Mapping (Hungarian Algorithm)
    # linear_sum_assignment minimizes cost, so we pass negative confusion matrix.
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create the mapping dictionary: {Alt_State: Baseline_State}
    # col_ind is the state in Alt, row_ind is the matching state in Baseline
    mapping = {alt: base for base, alt in zip(row_ind, col_ind)}
    
    # 3. Remap the Alternative States to match Baseline
    remapped_alt = np.array([mapping[s] for s in alternative_states])
    
    # 4. Calculate Agreement on REMAPPED states
    agreement = np.mean(baseline_states == remapped_alt)
    ari = adjusted_rand_score(baseline_states, remapped_alt)
    
    return {
        "agreement_pct": agreement,
        "ari_score": ari,
        "remapped_states": remapped_alt
    }

def run_sensitivity_suite(X, baseline_states, n_states, n_trials=10):
    results = []
    stability_matrix = np.zeros((len(X), n_trials))
    
    print(f"Starting Sensitivity Suite: {n_trials} trials...")
    
    # Trial 1-5: Random Seeds
    for i in range(5):
        alt_states = fit_perturbed_hmm(X, n_states, random_state=i+42)
        metrics = calculate_stability_metrics(baseline_states, alt_states)
        results.append({
            "type": "seed_variation", 
            "trial": i, 
            "agreement": metrics["agreement_pct"],
            "ari": metrics["ari_score"]
        })
        stability_matrix[:, i] = metrics["remapped_states"]

    # Trial 6-10: Noise Injection
    for i in range(5):
        X_noisy = perturb_data(X, noise_level=0.05)
        alt_states = fit_perturbed_hmm(X_noisy, n_states, random_state=1)
        metrics = calculate_stability_metrics(baseline_states, alt_states)
        results.append({
            "type": "noise_injection", 
            "trial": i, 
            "agreement": metrics["agreement_pct"],
            "ari": metrics["ari_score"]
        })
   
        stability_matrix[:, i+5] = metrics["remapped_states"]
        
    return pd.DataFrame(results), stability_matrix