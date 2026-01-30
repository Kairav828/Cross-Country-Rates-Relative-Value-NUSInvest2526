# Pair Eligibility Selection Logic

## Purpose
Define the eligibility rules that determine which country pairs are allowed to proceed to cointegration testing. This file defines decision gates and evidence requirements. It does not lock a final pair universe.

## Core principle
A spread is a valid relative value (RV) candidate only if:
- the relationship is stable (not fragile to time window choice),
- the spread is not a disguised global directional bet, and
- the underlying factor structure does not rotate frequently (structural break risk).

## Required inputs
- Δyields panel (same tenor across countries; PCA uses 5Y initially)
- Rolling PCA outputs:
  - `results/pca_5Y_fullsample_loadings.csv`
  - `results/pca_5Y_rolling_pc1_metrics.csv`
  - `results/pca_5Y_rolling_pc1_loadings.csv`
- Rolling correlation and clustering outputs

## Gate 1 — Rolling correlation stability (pair-level)
### Question
Does the pair exhibit stable co-movement over time (especially in calm regimes)?

### Evidence
- Rolling correlation of Δyields for the pair (window consistent with PCA rolling window)
- Summary metrics:
  - mean rolling correlation
  - rolling correlation volatility (std)
  - fraction of sign flips

### Fail conditions (typical)
- frequent sign flips
- low mean correlation combined with high instability

### Notes
Correlation is necessary but not sufficient. High correlation can be caused by shared global exposure.

## Gate 2 — Clustering persistence (system-level)
### Question
Do both countries belong to the same structural bloc consistently (or predictably by regime)?

### Evidence
- Rolling clustering assignments on Δyields (or PCA loadings)
- Persistence metric:
  - fraction of windows where both countries appear in the same cluster

### Fail conditions (typical)
- persistent fragmentation with no regime coherence
- unstable cluster identity

### Notes
Clustering validates economic coherence at the bloc level.

## Gate 3 — PCA factor neutrality (global-directionality filter)
### Question
Is the spread materially exposed to PC1 (global rates factor)?

### Evidence
From `pca_5Y_fullsample_loadings.csv`:
- PC1 loading per country

### Operational metric
For a pair (i, j):
- `pc1_abs_sum = |loading_i| + |loading_j|`
- `pc1_same_sign = sign(loading_i) == sign(loading_j)`

### Interpretation
- High `pc1_abs_sum` and `pc1_same_sign = True` indicates disguised directionality.
- Lower PC1 exposure indicates improved RV purity.

## Gate 4 — Structural stability (eigenvector instability filter)
### Question
Does the dominant factor retain a stable identity across time?

### Evidence
From `pca_5Y_rolling_pc1_metrics.csv`:
- `pc1_cosine_to_fullsample`

From `pca_5Y_rolling_pc1_loadings.csv`:
- rolling PC1 loading paths per country

### Pass condition (typical)
- cosine similarity close to 1 for most windows, with instability concentrated in obvious stress periods

### Fail condition (typical)
- frequent cosine collapses (factor rotation), including during calm periods

## Decision taxonomy
- **Eligible**
  - Passes all gates, or has only mild regime-sensitive degradation that remains well-defined.
- **Conditionally eligible (regime-restricted)**
  - Fails in stress/high-vol regimes but passes cleanly in low-vol regimes.
- **Rejected**
  - Directional disguise (Gate 3), or structurally unstable (Gate 4), or fundamentally unstable co-movement (Gate 1/2 once available).

## Documentation requirement
For each pair that proceeds to cointegration testing:
- one sentence per gate explaining pass/conditional/fail
- citations to the relevant plot or metric file (Task 5 PCA CSVs; rolling correlation/clustering outputs when added)

## Output artifact (automated evidence)
- `results/pair_selection_metrics_5Y.csv` summarises PCA-based pair metrics and structural stability indicators.
- Rolling correlation and clustering persistence columns are included when those outputs exist.
