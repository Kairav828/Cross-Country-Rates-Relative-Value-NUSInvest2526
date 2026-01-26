import pandas as pd

def assert_stationarity_allowed(
    variables,
    stationarity_csv="results/stationarity_tests.csv",
    required_form="diff"
):
    """
    Prevents illegal statistical usage of non-stationary series.
    
    required_form:
        "diff"  → requires diff_stationarity == I(0)
        "level" → requires level_stationarity == I(0)
    """
    stationarity = pd.read_csv(stationarity_csv)

    failures = []

    for var in variables:
        row = stationarity[stationarity["column"] == var]
        if row.empty:
            failures.append(f"{var}: not found in stationarity matrix")
            continue

        if required_form == "diff":
            if "I(0)" not in row.iloc[0]["label"]:
                failures.append(f"{var}: Δseries not stationary")
        elif required_form == "level":
            if "I(0)" not in row.iloc[0]["label"]:
                failures.append(f"{var}: level not stationary")

    if failures:
        raise ValueError(
            "Stationarity guard failed:\n" + "\n".join(failures)
        )