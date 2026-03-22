"""
07b_hedge_effectiveness.py
C5 replacement -- Hedge Effectiveness Framework

Replaces unrealistic cost claims with rigorous hedge analysis:
    1. Rolling OLS hedge ratio (24M window) — shows time variation
    2. Hedge Effectiveness HE = 1 - Var(hedged)/Var(unhedged)
    3. Break-even basis risk analysis
    4. Sensitivity: hedge ratio +/- 20%

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c5b_hedge_effectiveness.csv
        results/tables/c5b_rolling_hedge_ratio.csv
        results/figures/fig_c5b_hedge_effectiveness.pdf
        results/figures/fig_c5b_rolling_hr.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

WINDOW = 24  # rolling window

HEDGES = [
    ("US_Steel_PPI",  "GR_Steel",       "Steel",    0.30),
    ("US_Fuel_PPI",   "GR_Fuel_Energy", "Fuel",     0.20),
    ("US_Brent",      "GR_General_Index","General",  None),
    ("US_Cement_PPI", "GR_Concrete",    "Cement",   0.30),
    ("US_PVC_PPI",    "GR_PVC_Pipes",   "PVC",      0.20),
]

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def ols_hedge_ratio(spot, futures):
    """Min-variance hedge ratio."""
    X = np.column_stack([np.ones(len(futures)), futures.values])
    y = spot.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1]


def hedge_effectiveness(spot, futures, h):
    """HE = 1 - Var(hedged)/Var(unhedged)."""
    hedged = spot - h * futures
    var_unhedged = np.var(spot)
    var_hedged = np.var(hedged)
    return 1 - var_hedged / var_unhedged if var_unhedged > 0 else 0


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape}")

    # ── Static hedge effectiveness ────────────────────────────────────────────
    he_rows = []
    for us_col, gr_col, label, weight in HEDGES:
        if us_col not in df.columns or gr_col not in df.columns:
            continue

        h = ols_hedge_ratio(df[gr_col], df[us_col])
        he = hedge_effectiveness(df[gr_col], df[us_col], h)

        # Sensitivity: h +/- 20%
        he_low  = hedge_effectiveness(df[gr_col], df[us_col], h * 0.8)
        he_high = hedge_effectiveness(df[gr_col], df[us_col], h * 1.2)

        # Crisis only
        crisis = df.loc["2021-01-01":"2024-12-01"]
        if gr_col in crisis.columns and us_col in crisis.columns:
            h_crisis = ols_hedge_ratio(crisis[gr_col], crisis[us_col])
            he_crisis = hedge_effectiveness(crisis[gr_col], crisis[us_col], h_crisis)
        else:
            h_crisis, he_crisis = 0, 0

        he_rows.append({
            "Pair": label,
            "US_instrument": us_col,
            "GR_exposure": gr_col,
            "Weight": weight,
            "Hedge_Ratio_full": round(h, 4),
            "HE_full_%": round(he * 100, 1),
            "HE_sensitivity_low_%": round(he_low * 100, 1),
            "HE_sensitivity_high_%": round(he_high * 100, 1),
            "Hedge_Ratio_crisis": round(h_crisis, 4),
            "HE_crisis_%": round(he_crisis * 100, 1),
        })
        print(f"  {label:10s}  h={h:.4f}  HE={he*100:.1f}%  "
              f"crisis h={h_crisis:.4f}  HE_crisis={he_crisis*100:.1f}%")

    he_df = pd.DataFrame(he_rows)
    he_df.to_csv(os.path.join(OUT_TAB, "c5b_hedge_effectiveness.csv"), index=False)

    # ── Rolling hedge ratio ───────────────────────────────────────────────────
    rolling_rows = []
    fig, axes = plt.subplots(len(HEDGES), 1, figsize=(10, 2.5 * len(HEDGES)),
                              sharex=True)

    for idx, (us_col, gr_col, label, weight) in enumerate(HEDGES):
        if us_col not in df.columns or gr_col not in df.columns:
            continue

        hrs, hes, dates = [], [], []
        for i in range(WINDOW, len(df)):
            spot = df[gr_col].iloc[i - WINDOW:i]
            fut  = df[us_col].iloc[i - WINDOW:i]
            h = ols_hedge_ratio(spot, fut)
            he = hedge_effectiveness(spot, fut, h)
            hrs.append(h)
            hes.append(he)
            dates.append(df.index[i])

        rolling_rows.extend([{
            "Date": d, "Pair": label,
            "Hedge_Ratio": round(h, 4), "HE_%": round(he * 100, 1)
        } for d, h, he in zip(dates, hrs, hes)])

        ax = axes[idx]
        ax.plot(dates, hrs, color="#1565C0", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"{label}\nh(t)", fontsize=8)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5b_rolling_hr.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c5b_rolling_hr.pdf saved.")

    pd.DataFrame(rolling_rows).to_csv(
        os.path.join(OUT_TAB, "c5b_rolling_hedge_ratio.csv"), index=False)

    # ── HE bar chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(he_df))
    w = 0.35
    ax.bar(x - w/2, he_df["HE_full_%"], w, label="Full period",
           color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, he_df["HE_crisis_%"], w, label="Crisis (2021-24)",
           color="#D84315", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(he_df["Pair"])
    ax.set_ylabel("Hedge Effectiveness (%)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5b_hedge_effectiveness.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c5b_hedge_effectiveness.pdf saved.")
    print(f"\nSaved: c5b_hedge_effectiveness.csv, c5b_rolling_hedge_ratio.csv")


if __name__ == "__main__":
    main()
