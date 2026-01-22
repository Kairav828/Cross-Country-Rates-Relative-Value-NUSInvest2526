# Variable Map

This table documents what each series represents, how it is treated (levels vs changes), and how it is used in the strategy pipeline.

| column                         | dataset     | category                   | level_vs_change   | intended_use                                                   | notes                           |
|:-------------------------------|:------------|:---------------------------|:------------------|:---------------------------------------------------------------|:--------------------------------|
| bond_yields__GTAUD10Y Govt     | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTAUD2Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTAUD5Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTCNY10Y Govt     | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTCNY2Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTCNY5Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTEUR10Y Govt     | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTEUR2Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTEUR5Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTJPY10Y Govt     | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTJPY2Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTJPY5Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTUSD10Y Govt     | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTUSD2Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| bond_yields__GTUSD5Y Govt      | bond_yields | Core market (rates)        | level (yields)    | Cointegration (levels); changes for correlation/PCA/clustering |                                 |
| cesi__CESIAUD Index            | cesi        | Policy & macro drivers     | level             | Macro surprise state; regime features                          |                                 |
| cesi__CESICNY Index            | cesi        | Policy & macro drivers     | level             | Macro surprise state; regime features                          |                                 |
| cesi__CESIEUR Index            | cesi        | Policy & macro drivers     | level             | Macro surprise state; regime features                          |                                 |
| cesi__CESIJPY Index            | cesi        | Policy & macro drivers     | level             | Macro surprise state; regime features                          |                                 |
| cesi__CESIUSD Index            | cesi        | Policy & macro drivers     | level             | Macro surprise state; regime features                          |                                 |
| policyrates__BOJDTR Index      | policyrates | Policy & macro drivers     | level             | Policy divergence; regime conditioning                         |                                 |
| policyrates__CHLRLPR1 Index    | policyrates | Policy & macro drivers     | level             | Policy divergence; regime conditioning                         |                                 |
| policyrates__EURR002W Index    | policyrates | Policy & macro drivers     | level             | Policy divergence; regime conditioning                         |                                 |
| policyrates__FDTR Index        | policyrates | Policy & macro drivers     | level             | Policy divergence; regime conditioning                         |                                 |
| policyrates__RBATCTR Index     | policyrates | Policy & macro drivers     | level             | Policy divergence; regime conditioning                         |                                 |
| dxy__BBDXY Index               | dxy         | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Broad USD funding stress proxy  |
| fx_1m__AUD1M BGN Curncy        | fx_1m       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_1m__CNH1M BGN Curncy        | fx_1m       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_1m__EUR1M BGN Curncy        | fx_1m       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_1m__JPY1M BGN Curncy        | fx_1m       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_ov_iv__AUDUSDVON BGN Curncy | fx_ov_iv    | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_ov_iv__EURUSDVON BGN Curncy | fx_ov_iv    | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_ov_iv__USDCNHVON BGN Curncy | fx_ov_iv    | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| fx_ov_iv__USDJPYVON BGN Curncy | fx_ov_iv    | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| move__MOVE Index               | move        | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Rates implied volatility proxy  |
| repo__CNRE07 Index             | repo        | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| repo__ESTR3MA Index            | repo        | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Short-term funding stress proxy |
| repo__MUTKCALM Index           | repo        | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      |                                 |
| repo__SOFRRATE Index           | repo        | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Short-term funding stress proxy |
| us_eq__SPX Index               | us_eq       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Risk-on / risk-off indicator    |
| us_eq__VIX Index               | us_eq       | Stress & regime indicators | level + change    | HMM regime inference; breakdown diagnostics; risk filters      | Equity risk regime proxy        |