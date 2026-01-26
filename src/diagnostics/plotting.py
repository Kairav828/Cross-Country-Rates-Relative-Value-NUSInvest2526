# NOTE: This is a visual diagnostic only.
# No statistical inference or stationarity assumptions are enforced here.

import matplotlib.pyplot as plt
import pandas as pd

def plot_structural_check(df, columns, title_prefix, window=60):
    """
    Standardized Triple-Plot for Structural Sanity Checks.
    Moved to src/ for reusability across notebooks.
    """
    for col in columns:
        plot_data = df[[col]].dropna()
        
        if plot_data.empty:
            print(f"Skipping {col}: No data found.")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{title_prefix}: {col}", fontsize=14, fontweight='bold')

        # 1. Levels
        axes[0].plot(plot_data.index, plot_data[col], color='#1f77b4', lw=1.5)
        axes[0].set_ylabel("Levels")
        axes[0].set_title("Drift Inspection", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 2. Diff
        diff = plot_data[col].diff()
        axes[1].plot(plot_data.index, diff, color='#2ca02c', alpha=0.7, lw=1)
        axes[1].set_ylabel("Daily Change")
        axes[1].set_title("Stability Inspection", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # 3. Vol
        vol = diff.rolling(window=window).std()
        axes[2].plot(plot_data.index, vol, color='#d62728', lw=1.5)
        axes[2].set_ylabel(f"{window}D Vol")
        axes[2].set_title("Vol Clustering", fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()