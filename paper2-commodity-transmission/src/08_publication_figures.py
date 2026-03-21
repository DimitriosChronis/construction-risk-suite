"""
08_publication_figures.py
Paper 2 -- Publication-ready figures for C1 + C2 + C6.

Generates from saved CSV results:
    fig1 - Kendall tau heatmap (10D cross-market)         [from 03]
    fig2 - Cross-correlation lag heatmap                   [from 04]
    fig3 - Rolling Kendall tau (12M window, 5 pairs)       [from 04]
    fig4 - IRF panel (all pairs)                           [from 04b]
    fig5 - FEVD bar chart                                  [from 04b]
    fig6 - Structural break: rolling lag + rho (Steel)     [from 04c]
    fig7 - EU robustness Granger bar chart                 [from 09]

Style: ASCE-compatible, no embedded titles, serif font, 300 DPI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(__file__)
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# ASCE style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.linewidth":  0.8,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.5,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


def fig1_kendall_heatmap():
    """10x10 Kendall tau heatmap (C1)."""
    path = os.path.join(OUT_TAB, "c1_kendall_matrix.csv")
    if not os.path.exists(path):
        print("SKIP fig1: c1_kendall_matrix.csv not found")
        return
    tau_mat = pd.read_csv(path, index_col=0)
    short = [c.replace("GR_", "GR:").replace("US_", "US:") for c in tau_mat.columns]
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(tau_mat, dtype=bool), k=1)
    sns.heatmap(tau_mat.values, mask=mask, annot=True, fmt=".2f",
                xticklabels=short, yticklabels=short,
                cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                square=True, linewidths=0.5, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig1_kendall_heatmap.pdf"))
    plt.close()
    print("fig1_kendall_heatmap.pdf saved.")


def fig2_lag_heatmap():
    """Cross-correlation lag heatmap (C2)."""
    path = os.path.join(OUT_TAB, "c2_crosscorr_lags.csv")
    if not os.path.exists(path):
        print("SKIP fig2: c2_crosscorr_lags.csv not found")
        return
    cc = pd.read_csv(path)
    pairs = cc["Pair"].unique()
    max_lag = cc["Lag"].max()
    mat = np.zeros((len(pairs), max_lag + 1))
    for i, pair in enumerate(pairs):
        sub = cc[cc["Pair"] == pair].sort_values("Lag")
        mat[i, :len(sub)] = sub["Tau"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(mat, annot=True, fmt=".3f",
                xticklabels=[f"{l}M" for l in range(max_lag + 1)],
                yticklabels=pairs, cmap="RdBu_r", center=0,
                vmin=-0.3, vmax=0.3, linewidths=0.5, ax=ax)
    ax.set_xlabel("Lag (US leads Greek by N months)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig2_lag_heatmap.pdf"))
    plt.close()
    print("fig2_lag_heatmap.pdf saved.")


def fig3_rolling_tau():
    """Rolling Kendall tau time series (C2)."""
    path = os.path.join(OUT_TAB, "c2_crosscorr_lags.csv")
    data_path = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                             "aligned_log_returns.csv")
    if not os.path.exists(data_path):
        print("SKIP fig3: aligned_log_returns.csv not found")
        return
    from scipy.stats import kendalltau
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    PAIRS = [
        ("GR_Steel",        "US_Steel_PPI",  "Steel"),
        ("GR_Fuel_Energy",  "US_Fuel_PPI",   "Fuel/Energy"),
        ("GR_Concrete",     "US_Cement_PPI", "Cement/Concrete"),
        ("GR_PVC_Pipes",    "US_PVC_PPI",    "PVC/Plastic"),
        ("GR_General_Index","US_Brent",      "General/Brent"),
    ]
    WINDOW = 12
    fig, ax = plt.subplots(figsize=(10, 4))
    for gr_col, us_col, label in PAIRS:
        if gr_col not in df.columns or us_col not in df.columns:
            continue
        taus, dates = [], []
        for i in range(WINDOW, len(df) + 1):
            tau, _ = kendalltau(df[gr_col].iloc[i - WINDOW:i],
                                df[us_col].iloc[i - WINDOW:i])
            taus.append(tau)
            dates.append(df.index[i - 1])
        ax.plot(dates, taus, label=label, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Rolling Kendall tau (12M window)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig3_rolling_tau.pdf"))
    plt.close()
    print("fig3_rolling_tau.pdf saved.")


def fig4_irf_panel():
    """IRF panel for all pairs (C2)."""
    path = os.path.join(OUT_TAB, "c2b_var_summary.csv")
    if not os.path.exists(path):
        print("SKIP fig4: c2b_var_summary.csv not found (run 04b first)")
        return
    # The IRFs are generated directly by 04b; this just checks existence
    irf_path = os.path.join(OUT_FIG, "fig_c2b_irf_all.pdf")
    if os.path.exists(irf_path):
        print("fig4: fig_c2b_irf_all.pdf already exists (from 04b).")
    else:
        print("fig4: Run 04b_var_irf.py to generate IRF figures.")


def fig5_fevd_bar():
    """FEVD bar chart at 12-month horizon (C2)."""
    path = os.path.join(OUT_TAB, "c2b_fevd_table.csv")
    if not os.path.exists(path):
        print("SKIP fig5: c2b_fevd_table.csv not found")
        return
    fevd = pd.read_csv(path)
    fevd12 = fevd[fevd["Horizon_months"] == 12]
    if fevd12.empty:
        print("SKIP fig5: no 12-month FEVD data")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fevd12["Pair"], fevd12["US_explains_%"],
            color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Variance of Greek series explained by US shocks (%)")
    for i, (pair, val) in enumerate(zip(fevd12["Pair"], fevd12["US_explains_%"])):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig5_fevd_bar.pdf"))
    plt.close()
    print("fig5_fevd_bar.pdf saved.")


def fig6_structural_break():
    """Structural break: Steel rolling lag + rho (C2)."""
    path = os.path.join(OUT_FIG, "fig_c2c_structural_break_steel.pdf")
    if os.path.exists(path):
        print("fig6: fig_c2c_structural_break_steel.pdf already exists (from 04c).")
    else:
        print("fig6: Run 04c_structural_break.py to generate.")


def fig7_eu_robustness():
    """EU robustness Granger bar chart (C6)."""
    path = os.path.join(OUT_TAB, "c6_eu_granger.csv")
    if not os.path.exists(path):
        print("SKIP fig7: c6_eu_granger.csv not found")
        return
    granger = pd.read_csv(path)
    if granger.empty:
        print("SKIP fig7: empty granger results")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    granger["Label"] = granger["US_var"] + " -> " + granger["EU_var"]
    colors = ["#4CAF50" if s else "#BDBDBD" for s in granger["Significant"]]
    ax.barh(granger["Label"],
            -np.log10(granger["Best_p"].clip(1e-10)),
            color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--",
               linewidth=1, label="p = 0.05")
    ax.set_xlabel("-log10(p-value)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig7_eu_robustness.pdf"))
    plt.close()
    print("fig7_eu_robustness.pdf saved.")


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    print("Generating Paper 2 publication figures...\n")
    fig1_kendall_heatmap()
    fig2_lag_heatmap()
    fig3_rolling_tau()
    fig4_irf_panel()
    fig5_fevd_bar()
    fig6_structural_break()
    fig7_eu_robustness()
    print("\nDone.")


if __name__ == "__main__":
    main()
