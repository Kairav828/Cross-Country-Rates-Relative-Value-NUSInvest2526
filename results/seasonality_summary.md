# Seasonality Detection Summary

## 1. Conclusion
* **Verdict:** Seasonality is **Weak** across both Funding and Core Markets.
* **Key Finding:** Volatility is driven by **Regimes** (e.g., 2022 Inflation) and **Idiosyncratic Shocks** (e.g., Sept 2019 Repo), not by recurring calendar months.
* **Action:** * **No Month-Dummy Variables:** We will NOT include month indicators in the model.
    * **Outlier Filtering:** We must tag/exclude the **September 2019** period in the Repo model to prevent it from skewing the "normal" spread baseline.

## 2. Evidence from Heatmaps

### A. USD Funding Stress (Repo / SOFR)
* **Visual Observation:** The heatmap is predominantly blue (low vol) with a single, massive red anomaly in **September 2019**.
* **Interpretation:** This corresponds to the 2019 Overnight Repo Crisis (liquidity crunch).
* **Seasonality Check:** Contrary to the "Year-End Turn" hypothesis, the December column (Month 12) is relatively calm in this dataset. The risk is **event-driven**, not seasonal.

### B. Bond Yields (US Govt 2Y)
* **Visual Observation:** The heatmap shows **horizontal bands** rather than vertical stripes.
    * **2008 & 2022:** Higher volatility (lighter colors/red spots) across most months.
    * **2012–2020:** Deep blue bands indicating the "Low Volatility / QE" regime.
* **Interpretation:** Volatility is determined by the macroeconomic environment of the *year*, not the specific *month*.
* **Statistical Implication:** "Year" or "Regime" is a predictive feature; "Month" is not.


## 3. Implication
* **Modeling Choice:** Overall seasonality is relatively weak and we will proceed with the the regime conditioned models 

* **Data Handling:** The Sept 2019 Repo data point is a structural break and will be treated as an outlier in the Relative Value training set.