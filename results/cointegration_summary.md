# Regime-Conditional Cointegration: Summary

**Generated:** 2026-02-17 02:01

## Overview

- **Pairs tested:** 4
- **Regimes tested:** 4 (States 0, 1, 2, 3)
- **Total static tests:** 16
- **ADF lag selection:** Schwert criterion
- **Rolling window:** 250 days

## Static Test Results by Regime

### State 0
- **Cointegrated pairs:** 0/4
- **Sample sizes:** 1453-1453 obs
- **Heteroskedasticity detected** in all pairs
- **Autocorrelation detected** in all pairs

### State 1
- **Cointegrated pairs:** 1/4
- **Sample sizes:** 1533-1536 obs
- **Heteroskedasticity detected** in all pairs
- **Autocorrelation detected** in all pairs

### State 2
- **Cointegrated pairs:** 0/4
- **Sample sizes:** 1485-1488 obs
- **Heteroskedasticity detected** in all pairs
- **Autocorrelation detected** in all pairs

### State 3
- **Cointegrated pairs:** 0/4
- **Sample sizes:** 364-364 obs
- **Heteroskedasticity detected** in all pairs
- **Autocorrelation detected** in all pairs

## Key Findings

### Regime Heterogeneity

Cointegration stability varies significantly across regimes:

- **State 0:** 0% pairs cointegrated
- **State 1:** 25% pairs cointegrated
- **State 2:** 0% pairs cointegrated
- **State 3:** 0% pairs cointegrated

### Diagnostic Checks

- **Heteroskedasticity:** 16/16 tests failed White test
- **Autocorrelation:** 16/16 tests failed Ljung-Box test

### Hedge Ratio Stability

- **USD-EUR 5Y:** β = 0.653 ± 0.082
- **USD-EUR 10Y:** β = 0.638 ± 0.043
- **JPY-AUD 5Y:** β = 0.235 ± 0.011
- **JPY-AUD 10Y:** β = 0.370 ± 0.018

## Recommendations

1. **Regime-specific trading rules:** Use different hedge ratios for each regime state
2. **Risk management:** Increase monitoring during regime transitions
3. **Model selection:** Consider regime-switching models for spread trading strategies