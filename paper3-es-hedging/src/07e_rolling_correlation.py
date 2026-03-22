"""
07e_rolling_correlation.py
Fix 4: Rolling Correlation Analysis for Hedge Pairs

Shows time-varying Pearson and Kendall correlations between GR-US pairs.
Demonstrates that correlations are unstable and collapse during crises,
providing analytical explanation for low hedge effectiveness.

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c5e_rolling_corr_summary.csv
        results/figures/fig_c5e_rolling_correlation.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

WINDOW = 24  # rolling window months

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",         "Steel",   "#1565C0"),
    ("US_Cement_PPI", "GR_Concrete",      "Cement",  "#D84315"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",   "Fuel",    "#2E7D32"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",     "PVC",     "#7B1FA2"),
]

CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape}")

    fig, axes = plt.subplots(len(PAIRS), 1,
                              figsize=(10, 2.5 * len(PAIRS)),
                              sharex=True)

    summary_rows = []

    for idx, (us_col, gr_col, label, color) in enumerate(PAIRS):
        if us_col not in df.columns or gr_col not in df.columns:
            continue

        # Rolling Pearson
        dates = []
        rho_pearson = []
        tau_kendall = []

        for i in range(WINDOW, len(df)):
            win_us = df[us_col].iloc[i - WINDOW:i].values
            win_gr = df[gr_col].iloc[i - WINDOW:i].values

            rho = np.corrcoef(win_us, win_gr)[0, 1]
            tau, _ = kendalltau(win_us, win_gr)

            dates.append(df.index[i])
            rho_pearson.append(rho)
            tau_kendall.append(tau)

        dates = pd.to_datetime(dates)
        rho_arr = np.array(rho_pearson)
        tau_arr = np.array(tau_kendall)

        # Full period stats
        rho_full = np.corrcoef(df[us_col].values, df[gr_col].values)[0, 1]
        tau_full, _ = kendalltau(df[us_col].values, df[gr_col].values)

        # Crisis stats
        crisis = df.loc[CRISIS_START:CRISIS_END]
        rho_crisis = np.corrcoef(crisis[us_col].values,
                                  crisis[gr_col].values)[0, 1]
        tau_crisis, _ = kendalltau(crisis[us_col].values,
                                    crisis[gr_col].values)

        # Correlation volatility (std of rolling rho)
        rho_vol = np.std(rho_arr)

        # Percentage of windows with rho > 0.5 (hedgeable)
        pct_hedgeable = np.mean(np.abs(rho_arr) > 0.5) * 100

        print(f"  {label:10s}: rho_full={rho_full:.3f}  "
              f"rho_crisis={rho_crisis:.3f}  "
              f"rho_vol={rho_vol:.3f}  "
              f"hedgeable={pct_hedgeable:.0f}%")

        summary_rows.append({
            "Pair": label,
            "rho_full": round(rho_full, 3),
            "rho_crisis": round(rho_crisis, 3),
            "rho_change": round(rho_crisis - rho_full, 3),
            "tau_full": round(tau_full, 3),
            "tau_crisis": round(tau_crisis, 3),
            "rho_rolling_std": round(rho_vol, 3),
            "rho_rolling_min": round(np.min(rho_arr), 3),
            "rho_rolling_max": round(np.max(rho_arr), 3),
            "pct_hedgeable_%": round(pct_hedgeable, 1),
        })

        # Plot
        ax = axes[idx]
        ax.plot(dates, rho_arr, color=color, linewidth=1, label="Pearson")
        ax.plot(dates, tau_arr, color=color, linewidth=0.7,
                linestyle="--", alpha=0.6, label="Kendall")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":",
                    label="HE=25% threshold")
        ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
                    alpha=0.15, color="#D84315")
        ax.set_ylabel(f"{label}\n" + r"$\rho(t)$", fontsize=8)
        ax.set_ylim(-0.6, 0.8)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5e_rolling_correlation.pdf"),
                bbox_inches="tight")
    plt.close()
    print("\nfig_c5e_rolling_correlation.pdf saved.")

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_TAB, "c5e_rolling_corr_summary.csv"),
                   index=False)
    print(f"Saved: c5e_rolling_corr_summary.csv")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
