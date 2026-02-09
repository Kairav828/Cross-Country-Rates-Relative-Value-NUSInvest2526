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
For an equal-weight spread S = y_i − y_j, the exposure to PC1 is proportional to:

pc1_exposure = |loading_i − loading_j|

Interpretation:
- Small |loading_i − loading_j| ⇒ spread cancels the global factor ⇒ more “pure RV”
- Large |loading_i − loading_j| ⇒ spread loads on global directionality ⇒ disguised directional bet

We additionally require this neutrality to hold in BOTH low-vol and high-vol PCA splits
(using:
- results/pca_5Y_pc1_loadings_low_vol.csv
- results/pca_5Y_pc1_loadings_high_vol.csv)

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

---

## Current eligibility decisions

See:
- results/pair_eligibility_table.csv
- results/pair_eligibility_summary.md

## Pair Eligibility Summary

### Global structural note (Gate 4)

- Rolling PCA PC1 cosine-to-fullsample: 0.569 of windows > 0.8. (Low values imply frequent eigenvector rotation.)

Interpretation: even if a pair is RV-neutral under full-sample PCA, frequent PC1 rotation means we must expect regime-dependent behaviour.

### Eligible pairs

- **USD-EUR (5Y)** — G1 corr(mean=0.46, std=0.16, p05=0.06); G2 cocluster=73.5%; G3 pc1_diff(full=0.044, low=0.007, high=0.047).
- **JPY-AUD (5Y)** — G1 corr(mean=0.30, std=0.12, p05=0.05); G2 cocluster=76.4%; G3 pc1_diff(full=0.052, low=0.002, high=0.086).

### Conditionally eligible pairs

- None


### Rejected pairs (reason codes)

- Any pair failing clustering persistence (Gate 2) is rejected (not the same structural bloc).

- Any pair failing PCA neutrality across regimes (Gate 3) is rejected (directional disguise risk).
