# Variable Map (How to Read It)

## What this file is
`variable_map.md` documents every column in `data/processed/master_df.parquet` and how we plan to use it in the strategy pipeline.

It answers 4 questions:
1. **What is this series?** (e.g., Japan 10Y yield, MOVE index, CESI)
2. **Which bucket does it belong to?** (rates vs macro vs stress)
3. **Do we use the series in levels, changes, or both?**
4. **Where does it get used in the pipeline?** (correlation/PCA, HMM regimes, cointegration, etc.)

## Column meanings
- **column**: The exact column name inside `master_df` (format: `{dataset}__{series_name}`).
- **dataset**: Which raw CSV the series came from (`bond_yields`, `policyrates`, `cesi`, `move`, etc.).
- **category**:
  - **Core market (rates)**: the trade universe (government yields).  
  - **Policy & macro drivers**: conditioning variables like policy rates and CESI.
  - **Stress & regime indicators**: risk proxies like MOVE, DXY, VIX, FX vols, repo/funding proxies.
- **level_vs_change**:
  - **level** = we use the raw level series directly.
  - **level + change** = we will use the level AND also compute changes/returns later (e.g. Δyields, ΔDXY).
  - Note: changes are not stored in `master_df` yet; they are computed downstream.
- **intended_use**:
  - **Correlation / clustering / PCA** uses *changes* (stationary-ish).
  - **Cointegration** uses *levels* (yields).
  - **HMM regimes** uses stress/macro indicators (often both levels and changes).
- **notes**: short interpretation if the ticker isn’t obvious.

## Why this matters
This prevents:
- mixing levels/changes incorrectly,
- treating a “relative value” trade as a hidden duration bet,
- or claiming a regime story without documenting what drives it.

If a variable is added/removed from `master_df`, this map should be regenerated.
