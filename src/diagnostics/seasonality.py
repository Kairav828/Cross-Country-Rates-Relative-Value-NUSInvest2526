import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_seasonality(series, name="Asset"):
    # ... (Keep this function exactly as it was) ...
    df = series.to_frame(name='value').copy()
    df['month'] = df.index.month
    df['is_year_end'] = df['month'].isin([12, 1])
    
    monthly_vol = df.groupby('month')['value'].std() * np.sqrt(252)
    
    ye_data = df[df['is_year_end']]['value'].dropna()
    rest_data = df[~df['is_year_end']]['value'].dropna()
    
    # Safety check for empty data
    if len(ye_data) < 2 or len(rest_data) < 2:
        return {
            "name": name, "monthly_vol": monthly_vol, "ye_ratio": 0,
            "p_value": 1.0, "significant": False
        }

    stat, p_value = stats.levene(ye_data, rest_data)
    is_significant = p_value < 0.05
    ratio = ye_data.std() / rest_data.std() if rest_data.std() != 0 else 0
    
    return {
        "name": name,
        "monthly_vol": monthly_vol,
        "ye_ratio": ratio,
        "p_value": p_value,
        "significant": is_significant
    }

def plot_seasonality_heatmap(df_changes, regex_filter, title):
    """
    Plots a Year vs Month heatmap. 
    UPDATED: Handles empty data gracefully.
    """
    cols = [c for c in df_changes.columns if regex_filter in c]
    if not cols:
        print(f" No columns found for filter: {regex_filter}")
        return

    target = cols[0]
    
    # FIX: Drop NaNs ONLY for this specific column, not the whole DF
    data = df_changes[target].dropna().copy()
    
    if data.empty:
        print(f" Skipping {title}: No valid data found after dropping NaNs.")
        return
    
    data = data.to_frame()
    data['year'] = data.index.year
    data['month'] = data.index.month
    
    pivot = data.pivot_table(index='year', columns='month', values=target, aggfunc='std')
    
    # Double check pivot isn't empty
    if pivot.empty or pivot.isnull().all().all():
        print(f"⚠️ Skipping {title}: Not enough data to create heatmap.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=False, cbar_kws={'label': 'Monthly Volatility'})
    plt.title(f"Seasonality Heatmap: {title} ({target})")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show() 