# Task 6 — Pair Eligibility Summary

## Global structural note (Gate 4)

- Rolling PCA PC1 cosine-to-fullsample: 0.569 of windows > 0.8. (Low values imply frequent eigenvector rotation.)

Interpretation: even if a pair is RV-neutral under full-sample PCA, frequent PC1 rotation means we must expect regime-dependent behaviour.

## Eligible pairs

- **USD-EUR (5Y)** — G1 corr(mean=0.46, std=0.16, p05=0.06); G2 cocluster=73.5%; G3 pc1_diff(full=0.044, low=0.007, high=0.047).
- **JPY-AUD (5Y)** — G1 corr(mean=0.30, std=0.12, p05=0.05); G2 cocluster=76.4%; G3 pc1_diff(full=0.052, low=0.002, high=0.086).

## Conditionally eligible pairs

- None


## Rejected pairs (reason codes)

- Any pair failing clustering persistence (Gate 2) is rejected (not the same structural bloc).

- Any pair failing PCA neutrality across regimes (Gate 3) is rejected (directional disguise risk).
