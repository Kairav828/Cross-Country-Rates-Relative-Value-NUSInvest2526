# Stationarity Decisions

## Summary
Test stationarity using ADF (unit-root null) and KPSS (stationarity null) on:
- levels
- first differences

We classify a series as I(0) if:
- ADF p < 0.05 AND KPSS p > 0.05

We classify a series as I(1)-like if:
- ADF p >= 0.05 AND KPSS p <= 0.05

Ambiguous cases are treated conservatively.

---

## Rule 1 — Yield Levels vs Yield Changes
- Yield levels (bond_yields__*) are predominantly I(1)-like.
- Yield changes are predominantly stationary.

**Decision**
- Yield changes → correlation, clustering, PCA
- Yield levels → cointegration only
- Trading requires stationary residual spreads

---

## Rule 2 — Macro and Policy Variables
- Policy rates and CESI are persistent and often non-stationary in levels.

**Decision**
- Use as regime features or conditioning variables
- Prefer changes or normalized deviations
- Never regress non-stationary macro levels on non-stationary yields

---

## Rule 3 — Stress Indicators
- MOVE, VIX, DXY, repo series are borderline in levels.

**Decision**
- Use changes, rolling z-scores, or volatility measures
- HMM features must be stationary

---

## Rule 4 — Forbidden Regressions
- Non-stationary Y on non-stationary X without cointegration
- Trending variables in OLS without transformation

---

## Key Modeling Rule
If stationarity is unclear, treat the series as non-stationary and transform it.
