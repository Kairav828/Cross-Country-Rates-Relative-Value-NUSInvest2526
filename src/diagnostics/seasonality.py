import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore



def analyze_seasonality(series_diff, name="Series"):

    # FIX 6:  Outlier Removal 
    #zscore > 6
    clean_series = series_diff.copy()
    z_scores = np.abs(zscore(clean_series, nan_policy='omit'))
    
    # Identify and remove outliers (preserving index for timing)
    outliers = z_scores > 6
    if outliers.sum() > 0:
        print(f"[{name}] Removed {outliers.sum()} outliers (Z-score > 6)")
        clean_series = clean_series[~outliers]

    # FIX #3: Precise Year-End Definition 
    # "Use Dec 15 – Jan 15 window"
    # Create a boolean mask for the 'Turn' period
    day = clean_series.index.day
    month = clean_series.index.month
    
    # Turn = (Dec & Day >= 15) OR (Jan & Day <= 15)
    is_turn = ((month == 12) & (day >= 15)) | ((month == 1) & (day <= 15))
    
    # Calculate Volatility Ratio
    vol_turn = clean_series[is_turn].std()
    vol_rest = clean_series[~is_turn].std()
    
    # Avoid division by zero
    ye_ratio = vol_turn / vol_rest if vol_rest != 0 else 0

    # FIX 4: Month Dummy Regression 
    # "regress change in series ~ month_dummies and report p-value"
    df_reg = clean_series.to_frame(name='val')
    df_reg['month'] = df_reg.index.month.astype(str) # Categorical
    
    # Create dummies (drop_first=True to avoid collinearity)
    X = pd.get_dummies(df_reg['month'], drop_first=True, dtype=int)
    X = sm.add_constant(X)
    Y = df_reg['val']
    
    try:
        model = sm.OLS(Y, X).fit()
        p_value = model.f_pvalue # The probability that ALL months are zero
    except Exception as e:
        print(f"Regression failed for {name}: {e}")
        p_value = 1.0

    
    # Significant if F-test passed OR Ratio is huge (> 1.5x)
    is_significant = (p_value < 0.05) or (ye_ratio > 1.5)

    return {
        "ye_ratio": ye_ratio,
        "p_value": p_value,
        "significant": is_significant,
        "vol_turn": vol_turn,
        "vol_rest": vol_rest
    }






def plot_seasonality_heatmap(data_input, regex_filter, title):

    #Change ----------------------------------------------------
    if isinstance(data_input, pd.Series):
        
        target = regex_filter  # Use the filter name as the label
        data = data_input.dropna().copy()
    else:
        # If input is a DataFrame, search for the column
        cols = [c for c in data_input.columns if regex_filter in c]
        if not cols:
            print(f" No columns found for filter: {regex_filter}")
            return
        target = cols[0]
        data = data_input[target].dropna().copy()
    #Change end ------------------------------------------------
    if data.empty:
        print(f" Skipping {title}: No valid data found after dropping NaNs.")
        return
    
    # Convert to DataFrame for pivoting
    data = data.to_frame(name=target)
    data['year'] = data.index.year
    data['month'] = data.index.month
    
    pivot = data.pivot_table(index='year', columns='month', values=target, aggfunc='std')
    
    if pivot.empty or pivot.isnull().all().all():
        print(f" Skipping {title}: Not enough data to create heatmap.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=False, cbar_kws={'label': 'Monthly Volatility'})
    plt.title(f"Seasonality Heatmap: {title}")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()

