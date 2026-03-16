# What Regimes Explain (and Don’t)

## What regimes explain (causal story)
Our regime model is built to capture **macro + volatility state** that should plausibly drive:
- Cross-country co-movement strength
- Factor structure dominance (global vs relative factors)
- Market functioning (liquidity / flight-to-quality dynamics)

In short:
- **CALM umbrella (States 0/1/2)** → higher chance of stable relative relationships
- **STRESS umbrella (State 3)** → higher chance of fragmentation / structural breaks

These regimes are **not** derived from spreads, so they are eligible to be used as independent “environment labels” for next phase - cointegration + kalman filtering.

---

## What regimes do NOT explain (hard boundaries)
Regimes do not directly represent:
- Trade entry/exit timing
- Expected return / alpha
- Cointegration existence by themselves
- Optimal hedge ratios
- Spread mean reversion speed

Regimes are an **environment filter**, not a signal.

---

## What the current saved validation supports
### Supported by this phase's artifacts
- Economic interpretability: `results/regime_state_summary.csv`
- Persistence/structure of switching: `results/regime_transition_matrix.csv`, `results/regime_intervals.csv`
- Sensitivity analysis exists: `results/regime_stability_metrics.csv`

### Not fully supported in the saved hypothesis table
The file `results/regime_hypothesis_answers.csv` currently shows:
- Rolling correlation weakening in stress: **NOT AVAILABLE** (missing `rolling_corr_mean`)
- Cluster fragmentation in stress: **NOT AVAILABLE** (missing cluster metric)
- PC1 dominance in stress: **YES** (small but statistically significant difference)

So:
- We can responsibly claim: “Regimes are interpretable and robust-ish; PC1 dominance differs by regime.”
- We cannot yet claim: “Correlations/clusters clearly break in stress” using the saved summary outputs alone.

---

## How next phase should interpret this
Next phase should treat STRESS as:
- a **higher break-risk environment**
- a **place where cointegration must be re-tested**, not assumed
- a **regime where parameter instability is expected**

Next phase should treat CALM as:
- a **candidate environment** where stable relative value relationships are more plausible
- a place where conditional cointegration + time-varying hedges are expected to behave sensibly

---

## Known ambiguity zones (expected failure modes)
- Transition periods (probabilities in 0.40–0.60 band)
- Structural break episodes (e.g., 2008–09, 2020–03) where Gaussian emissions are strained
- Days where macro surprise is noisy but vol is calm (risk of “state split” among calm states)

These are are expected limitations of a Gaussian HMM on macro finance data.
