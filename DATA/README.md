# Data Directory

Raw data is NOT tracked in Git.

Expected structure:

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

Processed outputs (generated locally):
  processed/
    master_df.parquet

All raw data files are Bloomberg exports and must be obtained separately.