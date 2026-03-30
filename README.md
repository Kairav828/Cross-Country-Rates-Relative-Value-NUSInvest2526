# Cross-Country Rates Relative Value
### NUS Investment Society — Quantitative Research, Fixed Income | 2025/26

---

## Overview

This project builds a fully systematic, macro-conditioned cross-country rates relative value (RV) strategy on developed market sovereign yields. The central research question is:

> **When and why do cross-country interest rate relative value relationships hold — and when do they break?**

The answer depends on three conditions being simultaneously true: the spread must be statistically cointegrated within the current macro regime, the regime must not be one of abrupt structural breaks, and positions must be sized so that temporary adverse moves do not breach a hard stop before mean reversion occurs.

The strategy does not assume RV exists. It **proves when it exists**, **shows when it breaks**, and **trades it only when the underlying statistical and macro conditions jointly justify doing so**.

---

## Universe and Data

| Field | Detail |
|---|---|
| **Instruments** | 5Y and 10Y par sovereign yields: USD, EUR, JPY, AUD |
| **Sample** | January 2005 – December 2025 (monthly) |
| **Traded pairs** | USD–EUR 10Y, USD–EUR 5Y, JPY–AUD 10Y, JPY–AUD 5Y |

Pairs are selected dynamically each walk-forward window by a regime-conditional cointegration screen — not fixed in advance.

---

## Strategy Pipeline

```
── Exploratory Analysis ──────────────────────────────────────────────────────

NB01  Data diagnostics           (stationarity tests, yield level inspection)
NB02  Results and figures        (summary visuals)
NB03  Data visualisation         (yield time-series, spread charts)
NB04  Seasonality detection      (calendar effects in yield changes)
NB05  PCA structure              (global factor decomposition)
NB06  Rolling correlation        (time-varying cross-market co-movement)
NB07  Clustering                 (k-means regime grouping)
NB08  Regime robustness          (stability checks on cluster assignments)

── Active Strategy Framework ─────────────────────────────────────────────────

NB09  Macro feature engineering  (4 features used in NB17: PC1 slope, MOVE change, BBDXY change, Citi CESI PC1)
  ↓
NB10  HMM regime estimation      (Gaussian HMM, BIC model selection k = 2–4)
  ↓
NB11  Regime validation          (persistence, economic coherence, event cross-check)
  ↓
NB12  Regime-conditional cointegration screen
                                 (Fisher ADF, AR(1) mean reversion, half-life filter)
  ↓
NB17  Walk-forward window generation
                                 (expanding train windows, Kalman dynamic hedge ratio,
                                  innovation_z_t signal, regime labels for test windows)
  ↓
NB18  Fixed-parameter backtest   (Z_entry = 2.0, Z_exit = 0.5)
NB19  Walk-forward PnL-optimised (grid search, training PnL objective)
NB20  Walk-forward Sharpe-optimised (grid search, training Sharpe objective)
  ↓
NB21  Comprehensive performance analysis
                                 (regime-conditional PnL, robustness sweeps, failure case mapping)
NB22  Final summary write-up     (research question, methodology, findings, failure modes, extensions)

── Note ──────────────────────────────────────────────────────────────────────

NB13–16 are exploratory prototypes and are NOT part of the active framework.
NB17 supersedes all of their methods:
  NB13 (OLS/static hedge ratio)    →  NB17 uses Kalman dynamic hedge ratio
  NB14 (OU fitting on full sample) →  NB17 tests mean reversion regime-conditionally
  NB15 (signal design prototype)   →  NB17 generates innovation_z_t walk-forward
  NB16 (single-pass backtest)      →  NB18–20 are the production backtest engines
Where any method appears in both NB17 and NB13–16, NB17 is authoritative.
```

---

## Methodology

### 1. Macro Regime Identification (HMM)

A Gaussian Hidden Markov Model is fit on 4 macro features per period: PC1 of yield curve slope, MOVE change, BBDXY change, and PC1 of the Citi Economic Surprise Index (EUR & US). BIC selects the number of states from {2, 3, 4}. Identified states correspond to economically interpretable regimes: low-vol carry, risk-off flight-to-quality, policy divergence, and crisis.

HMMs are fit exclusively on training data. Test-window regime labels are inferred via Viterbi on frozen training parameters — no lookahead.

### 2. Regime-Conditional Cointegration Screen

For each (pair, regime) combination in the training window, three conditions must hold simultaneously:

| Condition | Test | Threshold |
|---|---|---|
| Spread stationarity | Fisher combined ADF | p < 0.05 |
| Mean reversion | AR(1) coefficient alpha | alpha < 0 |
| Half-life viability | P(half-life <= window length) | > 0.40 |

Only validated (pair, regime) combinations are tradeable. This rejects positions during regimes where the cointegrating anchor is statistically unreliable.

### 3. Kalman Filter Dynamic Hedge Ratio

The yield spread relationship is modelled as a state-space system with time-varying hedge ratio beta_t and intercept alpha_t. Process noise delta = 1e-6 keeps the hedge ratio slow-drifting — responsive to structural shifts but regularised against noise.

The trading signal is the **Kalman innovation** — the one-step-ahead prediction residual standardised by its predicted variance:

```
innovation_z_t  =  v_t / sqrt(S_t)

where:
  v_t  =  y_t  -  (beta_hat_{t|t-1} * x_t  +  alpha_hat_{t|t-1})
  S_t  =  H * P_{t|t-1} * H'  +  R
```

Under a correctly specified model this is white noise. A persistent deviation signals that the spread has dislocated beyond what the current macro regime justifies.

### 4. Regime-Gated Signal and Hard Stop

```
Enter long spread:   innovation_z_t  <  -Z_entry   (spread cheap vs model)
Enter short spread:  innovation_z_t  >  +Z_entry   (spread rich vs model)
Exit:                |innovation_z_t| <   Z_exit
Hard stop:           |innovation_z_t| >=  3.5       (independent of optimizer)
```

The hard stop is calibrated from the Max Adverse Excursion (MAE) distribution across all trades: p90 of MAE on winning trades = 2.95, p95 = 4.14. Threshold set at 3.5 — preserving ~90% of winning trades while preventing runaway losses. It is fixed before the optimizer runs and never tuned.

Exit priority: Hard Stop > Regime Shift (Forced) > Mean Reversion > End-of-Period MTM.

### 5. Walk-Forward Backtesting

Expanding training windows from 2005, with 1-year test steps (15 windows: 2010–2025). Every parameter (HMM, cointegration filter, Kalman hedge ratio) is estimated on training data only and applied forward. No refitting mid-window.

Three engines test different parameter selection approaches:
- **NB18** — fixed Z_entry = 2.0, Z_exit = 0.5
- **NB19** — walk-forward grid search, training PnL objective
- **NB20** — walk-forward grid search, training Sharpe objective

---

## Results

All performance is quoted in bps of yield spread PnL. Notional convention: USD 1MM DV01 (1 bp = USD 1,000).

| Engine | Trades | Win Rate | Ann. Sharpe | Total PnL | Max Drawdown | Hard Stops |
|---|---|---|---|---|---|---|
| **NB18 Fixed** | 56 | 64.3% | 0.63 | +8,955 bps | −1,849 bps | 6 |
| **NB19 WF PnL** | 121 | 65.3% | 0.80 | +15,220 bps | −3,453 bps | 7 |
| **NB20 WF Sharpe** | 121 | 64.5% | 0.77 | +14,487 bps | −3,453 bps | 7 |

*15 years out-of-sample walk-forward (2010–2025), 4 pairs, 5 bps round-trip transaction cost*

**Key findings:**

- Win rates are consistent at 64–65% across all three engines — the edge is in the signal, not the parameter selection.
- Profit factors of 2.30–3.13: average wins are 1.2–1.4x the magnitude of average losses.
- The walk-forward optimizer consistently selects Z_entry = 1.5, doubling trade count and total PnL vs the fixed engine but also doubling max drawdown.
- Hard stops are rare (~5% of trades) but represent real macro dislocation events — regime-lag entries that failed to mean-revert.
- Strategy remains Sharpe-positive at transaction costs up to ~12 bps (vs 5 bps assumed).

---

## Failure Modes

| Failure Mode | Mechanism |
|---|---|
| **HMM detection lag** | Model assigns prior regime label for 1–3 months during fast dislocations (COVID March 2020, GFC 2008). Hard stop limits damage. |
| **XCCY basis blowouts** | Cross-currency basis not modelled. During USD funding stress, the real cost of holding cross-currency positions exceeds the modelled 5 bp flat assumption. |
| **Structural policy breaks** | BOJ YCC (2016–2023), ECB negative rates (2015–2022) permanently relocate yield anchors. Cointegration filter catches these in-sample but transition periods produce false entries. |
| **Sparse trade count** | 56–121 trades over 15 years. Performance estimates carry wide confidence intervals — insufficient sample for robust statistical inference. |

---

## Methods

- Time-series diagnostics: ADF, KPSS, Fisher combined p-value
- Rolling correlation and k-means clustering
- Principal Component Analysis (global factor decomposition)
- Gaussian Hidden Markov Models with BIC model selection
- Engle–Granger cointegration, AR(1) half-life estimation
- Kalman filter state-space model (time-varying hedge ratio)
- Regime-conditional cointegration screening
- Walk-forward backtesting with expanding windows
- Max Adverse Excursion calibration for hard stop

No price prediction models. No black-box ML. Every decision is statistically grounded and macro-justified.

---

## Repository Structure

```
NOTEBOOKS/    Reproducible analysis (NB01–NB12, NB17–NB22; NB13–16 are exploratory only)
src/          Modular research and trading code
results/      Output CSVs (regime probabilities, trade ledgers, performance)
DATA/         Cleaned inputs (raw data excluded)
docs/         Strategy documentation
```

---

## Key Takeaway

Rate RV relationships hold when policy cycles are correlated, funding is benign, and yield curves are in a stable low-vol carry regime. They break when the macro regime shifts faster than the model can detect, or when a structural policy change permanently relocates the spread anchor. The regime gate handles most of the latter. The hard stop handles the residual. The result is consistent positive out-of-sample Sharpe across all three engines over 15 years.
