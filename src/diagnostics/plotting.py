# NOTE: This is a visual diagnostic only.
# No statistical inference or stationarity assumptions are enforced here.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_structural_check(df, columns, title_prefix, window=60):
    # Plots of drift, difference (Stationarity), and 60d rolling vol

    for col in columns:
        plot_data = df[[col]].dropna().copy()

        if plot_data.empty:
            print(f"skipping {col}: no valid plot_data after dropping NaNs")
            continue
        
        fig,axes = plt.subplots(3,1, figsize = (12, 10), sharex = True)
        fig.suptitle(f"{title_prefix}: {col}", fontsize = 16)

        #1st plot - Drift
        axes[0].plot(plot_data.index, plot_data[col], color = 'blue', lw = 1.5)
        axes[0].set_ylabel('Levels')
        axes[0].set_title(f"Level Inspection (Drift)")
        axes[0].grid(True)

        #2nd plot - Difference
        diff = plot_data[col].diff()
        axes[1].plot(plot_data.index, diff, color='#2ca02c', alpha=0.7, lw=1)
        axes[1].axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5) 
        axes[1].set_ylabel("Daily Change")
        axes[1].set_title("Stability Inspection (Stationarity)", fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        #3rd plot - Rolling vol (visualize regime clusters)
        vol = diff.rolling(window=window).std()
        
        axes[2].plot(plot_data.index, vol, color='#d62728', lw=1.5)
        axes[2].fill_between(plot_data.index, vol, color='#d62728', alpha=0.1)
        axes[2].set_ylabel(f"{window}D Vol")
        axes[2].set_title("Volatility Clustering (Regimes)", fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



def plot_structural_check_eq(df, columns, title_prefix, window=60):
    # for plotting of us_eq_indices, log returns used

    for col in columns:
        plot_data = df[[col]].dropna().copy()

        if plot_data.empty:
            print(f"skipping {col}: no valid plot_data after dropping NaNs")
            continue
        
        fig,axes = plt.subplots(3,1, figsize = (12, 10), sharex = True)
        fig.suptitle(f"{title_prefix}: {col}", fontsize = 16)

        #1st plot - Drift
        axes[0].plot(plot_data.index, plot_data[col], color = 'blue', lw = 1.5)
        axes[0].set_ylabel('Levels')
        axes[0].set_title(f"Level Inspection (Drift)")
        axes[0].grid(True)

        #2nd plot - Difference
        diff = 100 * np.log(plot_data[col] / plot_data[col].shift(1))
        axes[1].plot(plot_data.index, diff, color='#2ca02c', alpha=0.7, lw=1)
        axes[1].axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5) 
        axes[1].set_ylabel("Daily Change")
        axes[1].set_title("Stability Inspection (Stationarity)", fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        #3rd plot - Rolling vol (visualize regime clusters)
        vol = diff.rolling(window=window).std()
        
        axes[2].plot(plot_data.index, vol, color='#d62728', lw=1.5)
        
        axes[2].fill_between(plot_data.index, vol, color='#d62728', alpha=0.1)
        axes[2].set_ylabel(f"{window}D Vol")
        axes[2].set_title("Volatility Clustering (Regimes)", fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()