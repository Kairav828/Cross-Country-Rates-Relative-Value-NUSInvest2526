# Regression Checklist

## Purpose
This document defines the **statistical validity framework** for all regressions, time-series models, and statistical inferences in this project. It prevents spurious regressions, invalid coefficient estimates, and incorrect hypothesis tests.

**Before running any regression or time-series model, verify compliance with ALL applicable sections below.**

---

## 1. Stationarity Requirements

### 1.1 Core Principle
**OLS regression assumes both dependent and independent variables are stationary (I(0))**, unless cointegration applies.

I(0) -  Integrated of order 0: It means that the series does NOT need differencing to become stationary

### 1.2 Classification Rules
Based on ADF and KPSS tests (see [../results/stationarity_tests.csv](../results/stationarity_tests.csv)):

- **I(0) — Stationary**: ADF p < 0.05 AND KPSS p > 0.05
  - Safe to use in OLS directly
  - Examples: yield changes, CESI (some), VIX

- **I(1)-like — Non-stationary**: ADF p ≥ 0.05 AND KPSS p ≤ 0.05
  - **Cannot** be used in OLS without transformation or cointegration
  - Examples: yield levels, policy rates, DXY, repo rates

- **Ambiguous / Trend-stationary**: Mixed test results
  - **Treat as I(1) conservatively**
  - Examples: MOVE, some JPY yields, some FX vols

### 1.3 Required Transformations

| Variable Type | Stationarity Status | Valid Form for Regression |
|--------------|---------------------|---------------------------|
| **Bond yields** | I(1)-like in levels | Use Δyield (changes) OR cointegration |
| **Policy rates** | I(1)-like in levels | Use Δrate OR normalized deviations |
| **Macro (CESI)** | Mixed | Check test results; prefer changes if uncertain |
| **Stress (MOVE, VIX, DXY)** | Mixed/ambiguous | Use changes, rolling z-scores, or volatility measures |
| **Repo rates** | Mostly I(1)-like | Use changes OR spreads (if cointegrated) |
| **FX implied vol** | Ambiguous | Use changes OR volatility-of-volatility |

### 1.4 When Cointegration is Required
**If and only if** you need to model a **long-run equilibrium relationship between I(1) variables**:

**VALID**: Regress yield level A on yield level B **if** they are cointegrated (Engle-Granger or Johansen test confirms)
- The residual spread must be I(0)
- Example: Cross-country yield pair for relative value trading

**INVALID**: Regress I(1) yield level on I(1) policy rate without cointegration test
- Produces spurious R² and invalid t-stats

---

## 2. Autocorrelation

### 2.1 Definition
Correlation of a time series with a lagged version of itself, measuring how similar a variable's current values are to its past values. Ranges from [-1, 1].

### 2.2 Significance
Financial time series exhibit **serial correlation** in residuals, which:
- Underestimates standard errors
- Inflates t-statistics (false significance)
- Violates OLS assumptions

### 2.3 How to Test
Run **Durbin-Watson** or **Ljung-Box** test on regression residuals:
- DW statistic ≈ 2 suggests no autocorrelation
- DW < 1.5 or > 2.5 indicates autocorrelation
- Ljung-Box p < 0.05 rejects null of no autocorrelation

### 2.4 What to Do If Present

| Context | Solution |
|---------|----------|
| **Standard regression** | Use **Newey-West HAC** standard errors |
| **Time-series model** | Include lagged residuals (AR terms) or use ARIMA |
| **Panel regression** | Cluster standard errors by time or entity |
| **HMM / regime model** | Autocorrelation may be captured by states; verify residuals within-regime |

**WARNING**: Do NOT ignore autocorrelation. Report HAC-adjusted standard errors in all results.

---

## 3. Heteroskedasticity

### 3.1 Definition
Standard deviation of a variable monitored over time does not remain constant. Variance of errors (residuals) in a regression model is not constant across all levels of the independent variables.

### 3.2 Significance
Volatility clustering is pervasive in rates markets:
- Standard errors are inconsistent
- Hypothesis tests are invalid
- Does NOT bias coefficients, but affects inference

### 3.3 How to Test
- **Breusch-Pagan test**: Tests for systematic heteroskedasticity
- **White test**: General form (no specific structure assumed)
- **Visual**: Plot residuals vs fitted values; look for fan-shaped pattern

### 3.4 What to Do If Present

| Context | Solution |
|---------|----------|
| **Standard OLS** | Use **robust (White) standard errors** |
| **High volatility clustering** | Use **GARCH** or regime-conditional variance |
| **Crisis vs normal periods** | Estimate **subsample regressions** or use regime dummies |
| **Known structural breaks** | Split sample or include break dummies |

**DEFAULT**: Always report robust standard errors unless homoskedasticity is explicitly tested and confirmed.

---

## 4. Multicollinearity

### 4.1 Definition
Situation in multiple regression analysis where two or more independent variables are highly correlated.

### 4.2 Significance
Highly correlated regressors inflate coefficient variance:
- Coefficients become unstable
- Standard errors explode
- Individual t-stats are unreliable (even if F-stat is significant)

### 4.3 How to Test
Calculate **Variance Inflation Factor (VIF)** for each predictor:

$$
\text{VIF}_j = \frac{1}{1 - R^2_j}
$$

where $R^2_j$ is from regressing predictor $j$ on all other predictors.

**Thresholds**:
- VIF < 5: Acceptable
- VIF 5–10: Moderate multicollinearity; interpret with caution
- VIF > 10: **Severe**; action required

### 4.4 What to Do If Violated

| VIF Range | Action |
|-----------|--------|
| **5–10** | Report VIFs; acknowledge limitation; avoid over-interpreting individual coefficients |
| **> 10** | Remove one of the collinear variables OR use PCA/dimensionality reduction |
| **Policy rates + CESI** | Likely correlated; use **policy divergence** (rate spreads) instead |
| **Multiple stress indicators** | Use PCA or a composite stress index |

**WARNING**: Do NOT include 2Y, 5Y, 10Y yields from the same country as separate regressors without addressing multicollinearity (e.g., use curve slope instead).

---

## 5. Seasonality

### 5.1 When to Address
Seasonality must be modeled if:
- Year-end (Dec–Jan) volatility is **systematically higher** than rest-of-year
- Strong USD funding stress seasonality is present
- Variance ratio (Dec–Jan vs rest) > 1.5× with statistical significance

### 5.2 How to Test
- **Month dummy regression**: Regress changes on 11 month dummies; test joint significance
- **Variance comparison**: Compare variance(Dec–Jan) vs variance(Feb–Nov) using F-test
- **Seasonal decomposition**: Use STL or X-13 if quarterly patterns exist

### 5.3 What to Do

| Finding | Action |
|---------|--------|
| **No material seasonality** | Ignore; proceed with full-sample models |
| **Weak year-end effect** | Include December + January dummies in regressions |
| **Strong seasonal variance** | Estimate **regime-conditional models** (year-end vs rest) OR use rolling windows that avoid Dec–Jan |
| **Repo/funding seasonality** | Normalize repo spreads by within-month deviation |

**REQUIRED**: Document decision - state whether seasonality was tested and how it was handled in your analysis notes or project documentation.

---

## 6. Explicitly Forbidden Regressions

### 6.1 Spurious Regressions

**NEVER REGRESS**:
1. **I(1) yield level on I(1) macro level** without cointegration test
   - Example: `GTUSD10Y` ~ `FDTR` (both I(1)-like)
   - Why: Spurious correlation; coefficients are meaningless

2. **Non-stationary Y on non-stationary X** unless cointegration is established
   - Example: `GTEUR10Y` ~ `GTJPY10Y` without testing cointegration
   - Why: R² will be artificially high; t-stats invalid

3. **Changes on levels (or vice versa)** without theoretical justification
   - Example: `Δ bond_yields__GTUSD2Y` ~ `MOVE` (level)
   - Why: Mixing I(0) and I(1) variables is invalid unless MOVE is I(0) confirmed

### 6.2 Structural Issues

**NEVER DO**:
1. **Forward-looking data in explanatory variables**
   - Why: Causes look-ahead bias; destroys backtest validity

2. **Regress on lagged dependent variable without testing for unit root in residuals**
   - Example: $y_t = \alpha + \beta y_{t-1} + \epsilon_t$ with I(1) $y$
   - Why: This is nearly a random walk; coefficients are biased

3. **Use VIF > 10 variables together** without dimensionality reduction
   - Example: Including `GTUSD2Y`, `GTUSD5Y`, `GTUSD10Y` as separate regressors
   - Why: Unstable coefficients; cannot interpret individual effects

4. **Ignore heteroskedasticity in volatility-based models**
   - Example: Regressing spread changes on MOVE without robust SEs
   - Why: MOVE itself is volatile; residuals will be heteroskedastic

### 6.3 HMM / Regime Model Constraints

**NEVER USE**:
1. **Non-stationary features in HMM emissions**
   - Example: Feeding I(1) DXY level directly to HMM
   - Why: Assumes Gaussian emissions around stationary mean; violated by drift

2. **Overlapping regime indicators**
   - Example: Using both VIX and MOVE together without testing independence
   - Why: HMM may overfit; regimes become uninterpretable

**REQUIRED**: Transform all HMM features to stationary form (changes, z-scores, or volatility measures).

---

## 7. Valid Regression Templates

### 7.1 Correlation & PCA (OK)
```python
# Yield changes are I(0) → safe for correlation matrix
corr_matrix = master_df.filter(regex='bond_yields').diff().corr()
```

### 7.2 Spread Regression (OK if cointegrated)
```python
# Test cointegration first
residuals = ols(yield_A_level ~ yield_B_level).resid
if adf_test(residuals).pvalue < 0.05:
    # Spread is stationary → can use for trading
    spread = residuals
```
### 7.3 Macro Factor Regression (OK if stationary)
```python
# Use changes (I(0)) on both sides
ols(Δ yield ~ Δ policy_rate + Δ CESI + MOVE_change).fit(cov_type='HAC')
```

### 7.4 Regime-Conditional Model (OK)
```python
# HMM features must be I(0)
features = master_df[['DXY', 'MOVE', 'VIX']].diff().dropna()
hmm = GaussianHMM(n_components=3).fit(features)
```

## 8. Checklist Summary

Before running **any** regression or time-series model, confirm:

- [ ] All variables are I(0), OR cointegration is tested and confirmed
- [ ] Autocorrelation is tested; HAC standard errors used if present
- [ ] Heteroskedasticity is tested; robust standard errors used if present
- [ ] VIF < 10 for all regressors, OR multicollinearity is addressed
- [ ] Seasonality is tested; month dummies or regime-conditioning applied if material
- [ ] No forbidden regressions from Section 6 are attempted
- [ ] Results include diagnostic statistics (DW, VIF, robust SEs)

**If uncertain, default to conservative transformations (use changes, not levels).**

---

## 9. Interpretation & Reporting

### When presenting regression results, always include:
1. **Stationarity status** of all variables (cite [stationarity_tests.csv](../results/stationarity_tests.csv))
2. **Standard error type**: OLS, HAC, White, or clustered
3. **VIF values** if multiple regressors
4. **Durbin-Watson statistic** or Ljung-Box p-value for autocorrelation
5. **Sample period** and any subsample splits
6. **Regime conditioning** if applicable (e.g., "year-end excluded" or "HMM state 2 only")

### Example valid statement:
> "We regress 10Y–2Y USD curve slope changes (I(0), confirmed stationary) on CESI changes and MOVE changes (both I(0)). Newey-West standard errors with 5 lags are reported. VIF < 2 for all regressors. Durbin-Watson = 1.92 suggests no residual autocorrelation."

### Example invalid statement:
> **INVALID**: "We regress 10Y yield on Fed Funds rate. R² = 0.78, highly significant."
> - **Why invalid**: Both are I(1)-like; no cointegration test mentioned; no standard error adjustment.

---

## 10. Enforcement

This checklist is **mandatory** for:
- All regressions in Jupyter notebooks
- HMM feature engineering
- Cointegration-based spread construction
- Backtesting regression-based signals

**Peer review requirement**: Another team member must verify compliance before results are finalized.

**Interview-ready criterion**: You must be able to defend why each variable is in the form it is (level vs change, raw vs normalized) and what diagnostics were run.

---

## References
- [Stationarity test results](../results/stationarity_tests.csv)
- [Stationarity decisions](stationarity_decisions.md)
- [Variable map](variable_map.md)
- Engle & Granger (1987) on cointegration
- Newey & West (1987) on HAC standard errors
- Hamilton (1994) *Time Series Analysis* ch. 10-11 (spurious regression)

---

**Last updated**: 25 January 2026