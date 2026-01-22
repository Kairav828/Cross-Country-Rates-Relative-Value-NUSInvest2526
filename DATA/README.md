# Data Directory

Raw data is **NOT** tracked in Git.

## Expected structure

```text
DATA/
  raw/
    bond_yields.csv
    policyrates.csv
    citi_economic_surprise_index.csv
    OCE BofA MOVE INDEX.csv
    BBDXY.csv
    repo.csv
    us_equity_indicies.csv
    currency_1M_outright_normalizedtousdbaseccy.csv
    currency_Overnight_ATM_Implied_Vol.csv
```

Processed outputs (generated locally)
```text
DATA/
  processed/
    master_df.parquet
```

All raw data files are Bloomberg exports and must be obtained separately.

### Important details
- The folder tree is inside **```text** code fences → GitHub will preserve spacing perfectly.
- Separated “Expected structure” and “Processed outputs” into sections so it reads clean.

---

## Then commit just this file
From repo root:

```bash
git add DATA/README.md
git commit -m "Fix DATA README formatting"
git push
