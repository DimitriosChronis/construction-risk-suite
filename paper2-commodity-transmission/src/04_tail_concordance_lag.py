"""
04_tail_concordance_lag.py
C2 -- Greek ELSTAT vs Global Commodities: Tail Concordance Lag

Tests:
    1. Cross-correlation (Kendall tau) at lags 0..6 months
    2. Granger causality: Do US PPI indices Granger-cause Greek ELSTAT?
    3. Rolling Kendall tau (12M window) between matched pairs
    4. Tail concordance: joint exceedance probability at upper quantile

Key pairs:
    US_Steel_PPI -> GR_Steel
    US_Fuel_PPI  -> GR_Fuel_Energy
    US_Cement_PPI -> GR_Concrete
    US_PVC_PPI   -> GR_PVC_Pipes
    US_Brent     -> GR_General_Index

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c2_granger_causality.csv
        results/tables/c2_crosscorr_lags.csv
        results/figures/fig_c2_rolling_tau.pdf
        results/figures/fig_c2_lag_heatmap.pdf
"""

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

SEED     = 42
WINDOW   = 12     # rolling window months
MAX_LAG  = 6      # max lag for Granger test
ALPHA    = 0.05

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# Matched pairs: (Greek, US, label)
PAIRS = [
    ("GR_Steel",        "US_Steel_PPI",  "Steel"),
    ("GR_Fuel_Energy",  "US_Fuel_PPI",   "Fuel/Energy"),
    ("GR_Concrete",     "US_Cement_PPI", "Cement/Concrete"),
    ("GR_PVC_Pipes",    "US_PVC_PPI",    "PVC/Plastic"),
    ("GR_General_Index","US_Brent",      "General/Brent"),
]


def cross_corr_at_lags(x, y, max_lag):
    """Kendall tau at lags 0..max_lag (y leads x by lag months)."""
    results = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            tau, p = kendalltau(x, y)
        else:
            tau, p = kendalltau(x.iloc[lag:].values, y.iloc[:-lag].values)
        results.append({"Lag": lag, "Tau": round(tau, 4), "p_value": round(p, 4),
                        "Significant": p < ALPHA})
    return pd.DataFrame(results)


def rolling_kendall(x, y, window):
    """12-month rolling Kendall tau."""
    taus, dates = [], []
    for i in range(window, len(x) + 1):
        tau, _ = kendalltau(x.iloc[i - window:i], y.iloc[i - window:i])
        taus.append(tau)
        dates.append(x.index[i - 1])
    return pd.Series(taus, index=dates)


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Aligned data: {df.shape}")

    # ---- Cross-correlation at lags ----------------------------------------
    all_cc = []
    lag_matrix = np.zeros((len(PAIRS), MAX_LAG + 1))

    for i, (gr_col, us_col, label) in enumerate(PAIRS):
        if gr_col not in df.columns or us_col not in df.columns:
            print(f"  SKIP: {label}")
            continue
        cc = cross_corr_at_lags(df[gr_col], df[us_col], MAX_LAG)
        cc["Pair"] = label
        cc["Greek"] = gr_col
        cc["US"] = us_col
        all_cc.append(cc)

        best_lag = cc.loc[cc["Tau"].abs().idxmax(), "Lag"]
        best_tau = cc.loc[cc["Tau"].abs().idxmax(), "Tau"]
        lag_matrix[i, :] = cc["Tau"].values
        print(f"  {label:18s}  best lag={best_lag}M  tau={best_tau:+.3f}")

    cc_df = pd.concat(all_cc, ignore_index=True)
    cc_df.to_csv(os.path.join(OUT_TAB, "c2_crosscorr_lags.csv"), index=False)

    # ---- Lag heatmap -------------------------------------------------------
    labels = [p[2] for p in PAIRS]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(lag_matrix, annot=True, fmt=".3f",
                xticklabels=[f"{l}M" for l in range(MAX_LAG + 1)],
                yticklabels=labels, cmap="RdBu_r", center=0,
                vmin=-0.3, vmax=0.3, linewidths=0.5, ax=ax)
    ax.set_xlabel("Lag (US leads Greek by N months)")
    ax.set_ylabel("")
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c2_lag_heatmap.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c2_lag_heatmap.pdf saved.")

    # ---- Granger causality ------------------------------------------------
    granger_rows = []
    print("\nGranger causality tests (US -> Greek):")
    for gr_col, us_col, label in PAIRS:
        if gr_col not in df.columns or us_col not in df.columns:
            continue
        # Granger test: effect_col first, then cause_col
        data_pair = df[[gr_col, us_col]].dropna()
        try:
            res = grangercausalitytests(data_pair, maxlag=MAX_LAG, verbose=False)
            best_p = 1.0
            best_lag = 0
            for lag, test_dict in res.items():
                f_stat = test_dict[0]["ssr_ftest"][0]
                p_val  = test_dict[0]["ssr_ftest"][1]
                granger_rows.append({
                    "Pair": label, "Greek": gr_col, "US": us_col,
                    "Lag": lag, "F_stat": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "Significant": p_val < ALPHA
                })
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag
            sig = "YES ***" if best_p < 0.001 else "YES **" if best_p < 0.01 else "YES *" if best_p < 0.05 else "no"
            print(f"  {label:18s}  best p={best_p:.4f} at lag {best_lag}M  {sig}")
        except Exception as e:
            print(f"  {label:18s}  ERROR: {e}")

    granger_df = pd.DataFrame(granger_rows)
    granger_df.to_csv(os.path.join(OUT_TAB, "c2_granger_causality.csv"), index=False)

    # ---- Rolling tau plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    for gr_col, us_col, label in PAIRS:
        if gr_col not in df.columns or us_col not in df.columns:
            continue
        rtau = rolling_kendall(df[gr_col], df[us_col], WINDOW)
        ax.plot(rtau.index, rtau.values, label=label, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.4, color="orange", linestyle="--", linewidth=0.8)
    ax.axhline(-0.4, color="orange", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Rolling Kendall tau (12M window)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c2_rolling_tau.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c2_rolling_tau.pdf saved.")

    # ---- Summary -----------------------------------------------------------
    sig_count = granger_df[granger_df["Significant"]].groupby("Pair").size()
    print(f"\nGranger significant at any lag:")
    print(sig_count)


if __name__ == "__main__":
    main()
