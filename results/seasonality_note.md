# Seasonality Detection Note


All seasonality classifications are based on formally executed statistical tests rather than visual inspection.

1. Month-dummy regressions were run for each series, and joint significance was evaluated using the F-test. Series with p-values below 0.05 were classified as exhibiting statistically significant seasonal structure.

2. A complementary variance comparison was performed by computing the volatility ratio between the Dec–Jan window and the rest of the year (Feb–Nov). This quantifies whether year-end behavior differs materially in magnitude even if the regression test is not significant.

3. Decisions were made numerically:
   - Strong seasonality: statistically significant month effects (e.g., ESTR3MA with p = 0.0017).
   - Weak seasonality: insignificant p-values but elevated year-end variance ratios (e.g., CNRE07 ratio = 1.511, GTCNY2Y ratio = 2.085), handled as risk overlays rather than baseline model adjustments.
   - None: no statistical evidence of seasonality and variance ratios near 1, so effects are ignored.

This ensures handling rules are justified by test outputs rather than “eyeballing” patterns.


## repo__SOFRRATE Index
- YE variance ratio: 0.302
- F-test p-value: 0.9957
- Verdict: NONE
- Decision: Ignore seasonality

## repo__ESTR3MA Index
- YE variance ratio: 1.493
- F-test p-value: 0.0017
- Verdict: STRONG
- Decision: Add month dummies or treat Dec�Jan as separate regime

## repo__MUTKCALM Index
- YE variance ratio: 0.213
- F-test p-value: 0.2338
- Verdict: NONE
- Decision: Ignore seasonality

## repo__CNRE07 Index
- YE variance ratio: 1.511
- F-test p-value: 0.9783
- Verdict: WEAK
- Decision: Ignore in model but treat Dec�Jan as risk overlay

## move__MOVE Index
- YE variance ratio: 0.935
- F-test p-value: 0.3331
- Verdict: NONE
- Decision: Ignore seasonality

## dxy__BBDXY Index
- YE variance ratio: 0.949
- F-test p-value: 0.3288
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTUSD2Y Govt
- YE variance ratio: 0.902
- F-test p-value: 0.2917
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTUSD5Y Govt
- YE variance ratio: 0.906
- F-test p-value: 0.1747
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTUSD10Y Govt
- YE variance ratio: 0.939
- F-test p-value: 0.1706
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTEUR2Y Govt
- YE variance ratio: 0.834
- F-test p-value: 0.3463
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTEUR5Y Govt
- YE variance ratio: 0.82
- F-test p-value: 0.3634
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTEUR10Y Govt
- YE variance ratio: 0.889
- F-test p-value: 0.1349
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTJPY2Y Govt
- YE variance ratio: 0.736
- F-test p-value: 0.8494
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTJPY5Y Govt
- YE variance ratio: 0.876
- F-test p-value: 0.5899
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTJPY10Y Govt
- YE variance ratio: 1.02
- F-test p-value: 0.6281
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTCNY2Y Govt
- YE variance ratio: 2.085
- F-test p-value: 0.1999
- Verdict: WEAK
- Decision: Ignore in model but treat Dec�Jan as risk overlay

## bond_yields__GTCNY5Y Govt
- YE variance ratio: 1.017
- F-test p-value: 0.0696
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTCNY10Y Govt
- YE variance ratio: 0.809
- F-test p-value: 0.8675
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTAUD2Y Govt
- YE variance ratio: 0.902
- F-test p-value: 0.4217
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTAUD5Y Govt
- YE variance ratio: 0.997
- F-test p-value: 0.2055
- Verdict: NONE
- Decision: Ignore seasonality

## bond_yields__GTAUD10Y Govt
- YE variance ratio: 0.938
- F-test p-value: 0.2447
- Verdict: NONE
- Decision: Ignore seasonality

## fx_ov_iv__EURUSDVON BGN Curncy
- YE variance ratio: 1.173
- F-test p-value: 1.0
- Verdict: NONE
- Decision: Ignore seasonality

## fx_ov_iv__USDJPYVON BGN Curncy
- YE variance ratio: 1.103
- F-test p-value: 0.9991
- Verdict: NONE
- Decision: Ignore seasonality

## fx_ov_iv__USDCNHVON BGN Curncy
- YE variance ratio: 1.108
- F-test p-value: 1.0
- Verdict: NONE
- Decision: Ignore seasonality

## fx_ov_iv__AUDUSDVON BGN Curncy
- YE variance ratio: 0.871
- F-test p-value: 0.9985
- Verdict: NONE
- Decision: Ignore seasonality


