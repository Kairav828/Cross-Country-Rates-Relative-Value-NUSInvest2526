# Cross-Country Rates Relative Value 
# (NUS Investment Society Quantitative Research Department 25/26 - Fixed Income Team)

## Overview
This project develops and implements a macro-conditioned cross-country rates relative value trading framework.

The strategy identifies which government bond markets move together, determines when their relative value relationships are statistically valid, and trades regime-stable yield spreads using cointegration, Kalman filtering, and Ornstein–Uhlenbeck dynamics.

The core objective is **not unconditional profit maximisation**, but to systematically determine **when relative value assumptions hold, when they break, and how spread behaviour changes across macro regimes**, and to trade only when those assumptions are defensible.

---

## What This Project Does

### Identifies economically coherent cross-country rate relationships
- Focuses on Japan and Asia-Pacific markets (Korea, Australia, Singapore / China)
- Motivation:
  - trade linkages  
  - USD funding sensitivity  
  - global risk cycles  
  - Bank of Japan spillover effects  

### Discovers which countries move together and when
- Uses **yield changes** (stationary) for:
  - rolling correlations
  - clustering
  - PCA-based factor decomposition
- Detects bloc formation and fragmentation across macro conditions

### Separates directional risk from true relative value
- Uses PCA to ensure candidate spreads are factor-neutral
- Excludes “RV trades” that are actually disguised global duration bets

### Statistically infers macro regimes
- Hidden Markov Models classify regimes using:
  - rates volatility
  - inflation momentum
  - policy divergence
  - USD funding stress
- No manual regime labeling

### Constructs and validates tradeable spreads
- Tests yield levels for cointegration within regimes
- Only stationary residual spreads are retained
- Kalman filtering estimates:
  - time-varying hedge ratios
  - structural drift

### Trades regime-stable relative value spreads
- Models spreads as Ornstein–Uhlenbeck processes
- Uses:
  - innovation z-scores
  - half-life and reversion probabilities
  - regime probability gating
- Exits on:
  - mean reversion
  - regime breaks
  - time stops

### Evaluates performance conditionally
- Performance decomposed by:
  - volatility regime
  - inflation regime
  - funding stress
- Explicit analysis of failure cases and breakdowns
- Transaction cost sensitivity and walk-forward validation included

---

## Why This Is Different From Typical “RV Strategies”
- Relative value is treated as **conditional**, not universal
- Trading is disabled when macro or funding regimes invalidate equilibrium assumptions
- Emphasis is placed on:
  - stationarity
  - regression validity
  - regime stability
  - seasonality and balance-sheet constraints
- Trading results are used as **evidence**, not the starting point

This mirrors how institutional macro and relative value desks actually think about risk.

---

## Methods Used
- Time-series diagnostics (ADF, KPSS, residual tests)
- Rolling correlation and clustering
- Principal Component Analysis
- Hidden Markov Models
- Engle–Granger cointegration
- Kalman filtering
- Ornstein–Uhlenbeck modeling
- Data-driven threshold estimation
- Regime-conditional backtesting

No black-box price prediction models are used.

---

## Repository Structure
- src/ Modular research and trading code
- NOTEBOOKS/ Reproducible analysis and figures
- report/ Final written strategy report
- DATA/ Cleaned inputs (raw data excluded)
- tests/ Statistical validation checks

---

## Key Takeaway
This project does not assume relative value exists.

It **proves when it exists**, **shows when it breaks**, and **trades it only when the underlying statistical and macro conditions justify doing so**.
