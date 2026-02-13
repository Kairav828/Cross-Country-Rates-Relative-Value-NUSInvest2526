import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

"""
Regime Stability & Sensitivity

Purpose:
- Stress-test HMM regime assignments under:
  (1) different random seeds
  (2) noise injection to features

Outputs (returned by run_sensitivity_suite):
- results_df: per-trial stability metrics (agreement, ARI, flip-rate, durations)
- stability_matrix: (T x n_trials) matrix of remapped labels per trial
"""


# -----------------------------
# Helpers
# -----------------------------
def _run_lengths(labels: np.ndarray) -> np.ndarray:
    """Return run lengths for a 1D integer label array."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([], dtype=int)
    lengths = []
    run = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return np.array(lengths, dtype=int)


def _avg_duration_by_state(labels: np.ndarray) -> dict:
    """Average run length per state label."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}

    out = {}
    states = np.unique(labels)
    for s in states:
        s = int(s)
        runs = []
        run = 0
        for v in labels:
            if int(v) == s:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                    run = 0
        if run > 0:
            runs.append(run)

        out[f"avg_duration_state_{s}"] = float(np.mean(runs)) if runs else np.nan

    return out


def perturb_data(X: np.ndarray, noise_level: float = 0.01, random_state: int | None = None) -> np.ndarray:
    """Add IID Gaussian noise to X."""
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, noise_level, size=X.shape)
    return X + noise


def fit_hmm_states(X: np.ndarray, n_states: int, random_state: int | None = None) -> np.ndarray:
    """Fit a full-covariance Gaussian HMM and return Viterbi state path."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=random_state,
    )
    model.fit(X)
    return model.predict(X)


def calculate_stability_metrics(baseline_states: np.ndarray, alternative_states: np.ndarray) -> dict:
    """
    Compare baseline vs alternative regimes using optimal label matching (Hungarian).
    Returns agreement/ARI + flip-rate + duration metrics + remapped alternative path.
    """
    baseline_states = np.asarray(baseline_states).astype(int)
    alternative_states = np.asarray(alternative_states).astype(int)

    # Confusion matrix (rows=baseline, cols=alt)
    cm = confusion_matrix(baseline_states, alternative_states)

    # Hungarian mapping (maximize matches => minimize negative matches)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Mapping: alt_state -> baseline_state
    mapping = {int(alt): int(base) for base, alt in zip(row_ind, col_ind)}

    # Remap alt to baseline label space
    remapped_alt = np.array([mapping[int(s)] for s in alternative_states], dtype=int)

    # Core metrics
    agreement = float(np.mean(baseline_states == remapped_alt))
    ari = float(adjusted_rand_score(baseline_states, remapped_alt))

    # Required Task 3.4 metrics
    flip_rate = float(np.mean(remapped_alt != baseline_states))

    baseline_runs = _run_lengths(baseline_states)
    alt_runs = _run_lengths(remapped_alt)

    avg_duration_baseline = float(np.mean(baseline_runs)) if len(baseline_runs) else np.nan
    avg_duration_alt = float(np.mean(alt_runs)) if len(alt_runs) else np.nan

    per_state_baseline = _avg_duration_by_state(baseline_states)
    per_state_alt = _avg_duration_by_state(remapped_alt)

    return {
        "agreement": agreement,
        "ari": ari,
        "flip_rate": flip_rate,
        "avg_duration_baseline": avg_duration_baseline,
        "avg_duration_alt": avg_duration_alt,
        **{f"baseline_{k}": v for k, v in per_state_baseline.items()},
        **{f"alt_{k}": v for k, v in per_state_alt.items()},
        "remapped_states": remapped_alt,
    }


def run_sensitivity_suite(
    X: np.ndarray,
    baseline_states: np.ndarray,
    n_states: int,
    n_trials_seed: int = 5,
    n_trials_noise: int = 5,
    noise_level: float = 0.05,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run sensitivity tests:
      - seed variation: refit HMM with different seeds
      - noise injection: add noise then refit HMM

    Returns:
      - results_df: per-trial stability metrics
      - stability_matrix: (T x n_trials_total) remapped label paths
    """
    X = np.asarray(X)
    baseline_states = np.asarray(baseline_states).astype(int)

    n_trials_total = n_trials_seed + n_trials_noise
    stability_matrix = np.zeros((len(X), n_trials_total), dtype=int)
    results = []

    print(f"Starting Sensitivity Suite: {n_trials_total} trials...")

    # Trials 1..n_trials_seed: random seed variation
    for i in range(n_trials_seed):
        seed = 42 + i
        alt_states = fit_hmm_states(X, n_states=n_states, random_state=seed)
        metrics = calculate_stability_metrics(baseline_states, alt_states)

        results.append({
            "type": "seed_variation",
            "trial": i,
            "seed": seed,
            "noise_level": 0.0,
            **{k: v for k, v in metrics.items() if k != "remapped_states"},
        })
        stability_matrix[:, i] = metrics["remapped_states"]

    # Trials n_trials_seed..end: noise injection
    for j in range(n_trials_noise):
        idx = n_trials_seed + j
        X_noisy = perturb_data(X, noise_level=noise_level, random_state=100 + j)
        alt_states = fit_hmm_states(X_noisy, n_states=n_states, random_state=1)
        metrics = calculate_stability_metrics(baseline_states, alt_states)

        results.append({
            "type": "noise_injection",
            "trial": j,
            "seed": 1,
            "noise_level": float(noise_level),
            **{k: v for k, v in metrics.items() if k != "remapped_states"},
        })
        stability_matrix[:, idx] = metrics["remapped_states"]

    return pd.DataFrame(results), stability_matrix
