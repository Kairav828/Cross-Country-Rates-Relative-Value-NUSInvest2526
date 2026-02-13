
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

"""
Functions for validating regime assignments against structural metrics.

"""


def merge_regime_and_structure(df_regimes, df_structure):
    """
    Merges regime labels with structural metrics (PCA, Clusters) by Date.
    """
    # intersection of dates ensures we only analyze days where we have BOTH
    common_idx = df_regimes.index.intersection(df_structure.index)
    return df_regimes.loc[common_idx].join(df_structure.loc[common_idx])


def calculate_regime_stats(df_merged, structure_cols, regime_col='regime_most_likely'):
    """
    Calculates Mean, Std, and Count of structural metrics for each regime.
    """
    return df_merged.groupby(regime_col)[structure_cols].agg(['mean', 'std', 'count'])


def test_hypothesis(df_merged, metric, crisis_regime, normal_regime):
    """
    Performs a T-Test to check if a metric is significantly different 
    in the Crisis regime vs. the Normal regime.
    """
    crisis_data = df_merged[df_merged['regime_most_likely'] == crisis_regime][metric].dropna()
    normal_data = df_merged[df_merged['regime_most_likely'] == normal_regime][metric].dropna()
    
    # Welch's t-test (equal_var=False) is safer for financial data
    t_stat, p_val = ttest_ind(crisis_data, normal_data, equal_var=False)
    
    return {
        "metric": metric,
        "crisis_mean": crisis_data.mean(),
        "normal_mean": normal_data.mean(),
        "diff": crisis_data.mean() - normal_data.mean(),
        "p_value": p_val,
        "significant": p_val < 0.05
    }


def plot_regime_boxplots(df_merged, metrics):
    """
    Visualizes the distribution of metrics across regimes.
    """
    plt.figure(figsize=(15, 6))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        sns.boxplot(x='regime_most_likely', y=metric, data=df_merged, palette='viridis')
        plt.title(f"{metric} Distribution by Regime")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()