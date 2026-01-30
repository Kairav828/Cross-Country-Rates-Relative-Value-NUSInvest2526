# Seasonality Detection Summary

## 1. Conclusion

**Verdict:** Seasonality is weak across both Funding and Core Markets.

**Key Finding:** Volatility is primarily driven by policy regimes (e.g., 2022 inflation/hiking) and idiosyncratic shocks (e.g., Sept 2019 repo), rather than recurring calendar months.

**Action:**
- **No Month-Dummy Variables:** We will not include month indicators in the regression model as the F-tests failed to show statistical significance (p < 0.05).
- **Outlier Filtering:** We must surgically exclude the September 2019 (Repo Crisis) and March 2020 (COVID-19) periods to prevent these one-off structural breaks from skewing the baseline volatility.

---

## 2. Evidence from Heatmaps & Statistics

| Series                  | YE Vol Ratio (Dec/Jan vs Rest) | F-Test p-value | Verdict |
|-------------------------|--------------------------------|----------------|---------|
| USD Funding (Repo)      | 1.46x                          | 0.8122         | WEAK    |
| Bond Yields (US 10Y)    | 0.95x                          | 0.2576         | NONE    |
| Market Stress (MOVE)    | 1.01x                          | 0.3879         | NONE    |


### A. USD Funding Stress (Repo)

- **Visual Observation:** After removing the massive anomalies of 2019 and 2020, the heatmap reveals a subtle pattern of elevated volatility during the Dec/Jan window.
- **Statistical Reality:** The Year-End Volatility Ratio is 1.46x, confirming that balance sheet window-dressing creates a distinct liquidity constraint.
- **Interpretation:** While the “Year-End Turn” is economically real, the high F-test p-value (0.8122) indicates that over a 10-year sample, this seasonality is overshadowed by broader monetary policy regimes. It is a risk factor, but not a dominant predictive feature.

### B. Bond Yields (US Govt 10Y)

- **Visual Observation:** The heatmap displays horizontal bands rather than vertical stripes, indicating that volatility persists for entire years rather than specific months.
- **2008 & 2022:** Persistent high volatility (lighter colors) across all months.
- **2012–2020:** Deep blue bands indicating the “Low Volatility / QE” regime.
- **Statistical Reality:** The Year-End Volatility Ratio is 0.95x, effectively near parity.
- **Interpretation:** Volatility is determined by the macroeconomic environment of the regime, not the calendar month.

---

## 3. Implication

**Modeling Choice:** Given that seasonality is statistically weak compared to macro drivers, we will proceed with regime-conditioned models (focusing on volatility clusters) rather than calendar-based models.

**Tactical Note:** While excluded from the core model, the Year-End Turn (1.46x vol) should be treated as a risk overlay for short-term trading during the Dec 15–Jan 15 window.




