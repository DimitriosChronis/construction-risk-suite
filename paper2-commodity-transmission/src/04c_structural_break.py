"""
04c_structural_break.py
Structural Break Analysis (C2 Enhancement)

Tests whether US->GR commodity transmission changed around:
  Break 1: COVID-19  (2020-03)
  Break 2: Ukraine   (2022-02)

For each pair and break:
  1. Chow test (F-stat, p-value)
  2. Optimal transmission lag pre vs post break (cross-correlation)
  3. Pearson correlation change

Key hypothesis:
  Transmission lags SHORTENED and correlations INCREASED
  after COVID-19 supply-chain disruption (globalisation acceleration).

Outputs:
  results/tables/c2c_structural_breaks.csv
  results/figures/fig_c2c_structural_break_steel.pdf
  results/figures/fig_c2c_rolling_lag_all.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

BREAKS = {
    "COVID-19": "2020-03-01",
    "Ukraine":  "2022-02-01",
}

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",         "Steel"),
    ("US_Cement_PPI", "GR_Concrete",      "Cement"),
    ("US_Brent",      "GR_General_Index", "Brent"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",   "Fuel"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",     "PVC"),
]

MAX_LAG     = 6
ROLL_WINDOW = 36   # months


# ── Helpers ───────────────────────────────────────────────────────────────────

def ols_rss(y: np.ndarray, x: np.ndarray) -> float:
    X = sm.add_constant(x, has_constant="add")
    return float(np.sum(sm.OLS(y, X).fit().resid ** 2))


def chow_f(y: np.ndarray, x: np.ndarray, bp: int):
    """Chow test for structural break at index bp. Returns (F, p)."""
    n, k = len(y), 2          # k = intercept + slope
    if bp < k + 2 or (n - bp) < k + 2:
        return np.nan, np.nan
    rss_full = ols_rss(y, x)
    rss_pre  = ols_rss(y[:bp], x[:bp])
    rss_post = ols_rss(y[bp:], x[bp:])
    rss_pool = rss_pre + rss_post
    f  = ((rss_full - rss_pool) / k) / (rss_pool / (n - 2 * k))
    pv = float(1.0 - stats.f.cdf(f, k, n - 2 * k))
    return round(float(f), 3), round(pv, 4)


def best_lag(y: np.ndarray, x: np.ndarray, max_lag: int = MAX_LAG):
    """Return (lag, rho) with highest |cross-correlation| at lags 0..max_lag."""
    best_l    = 0
    best_r    = 0.0    # actual rho at best lag
    best_rabs = -1.0   # |rho| tracker — always < any real |rho| initially
    for lag in range(0, max_lag + 1):
        y_s = y[lag:] if lag > 0 else y
        x_s = x[:-lag] if lag > 0 else x
        if len(y_s) < 10:
            break
        r = float(np.corrcoef(y_s, x_s)[0, 1])
        if not np.isfinite(r):
            continue
        if abs(r) > best_rabs:
            best_rabs = abs(r)
            best_r    = r
            best_l    = lag
    return best_l, round(best_r, 3)


def stars(p) -> str:
    if np.isnan(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape[0]} obs x {df.shape[1]} series\n")

    rows = []

    for us_col, gr_col, label in PAIRS:
        if us_col not in df.columns or gr_col not in df.columns:
            print(f"  SKIP {label}: column not found")
            continue

        y = df[gr_col].values
        x = df[us_col].values

        lag_full, rho_full = best_lag(y, x)

        print(f"{'='*55}")
        print(f"{label:10s}  full: lag={lag_full}M  rho={rho_full:.3f}")

        for bname, bdate in BREAKS.items():
            bp = int(df.index.searchsorted(pd.Timestamp(bdate)))

            f_stat, p_val = chow_f(y, x, bp)
            sg = stars(p_val)
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "NA"
            print(f"  Chow ({bname:10s}): F={f_stat}  p={p_str} {sg}")

            lag_pre,  rho_pre  = best_lag(y[:bp], x[:bp])
            lag_post, rho_post = best_lag(y[bp:], x[bp:])

            print(f"    Pre-break:  lag={lag_pre}M  rho={rho_pre:.3f}  "
                  f"(n={bp})")
            print(f"    Post-break: lag={lag_post}M  rho={rho_post:.3f}  "
                  f"(n={len(y)-bp})")

            rows.append({
                "Pair":       label,
                "Break":      bname,
                "Chow_F":     f_stat if not np.isnan(f_stat) else "NA",
                "Chow_p":     p_val  if not np.isnan(p_val)  else "NA",
                "Stars":      sg,
                "Lag_full":   lag_full,
                "Rho_full":   rho_full,
                "Lag_pre":    lag_pre,
                "Rho_pre":    rho_pre,
                "Lag_post":   lag_post,
                "Rho_post":   rho_post,
                "Lag_change": lag_post - lag_pre,
                "Rho_change": round(rho_post - rho_pre, 3),
                "n_pre":      bp,
                "n_post":     len(y) - bp,
            })
        print()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT_TAB, "c2c_structural_breaks.csv"), index=False)

    # ── Rolling lag figure — Steel ─────────────────────────────────────────
    us_col, gr_col, label = PAIRS[0]   # Steel
    roll_dates, roll_lags, roll_rhos = [], [], []
    for i in range(ROLL_WINDOW, len(df)):
        y_w = df[gr_col].iloc[i - ROLL_WINDOW:i].values
        x_w = df[us_col].iloc[i - ROLL_WINDOW:i].values
        l_w, r_w = best_lag(y_w, x_w)
        roll_dates.append(df.index[i])
        roll_lags.append(l_w)
        roll_rhos.append(r_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(roll_dates, roll_lags, color="#1565C0", linewidth=1.2)
    ax1.set_ylabel("Optimal Transmission Lag (months)")
    ax1.set_yticks(range(0, MAX_LAG + 1))

    ax2.plot(roll_dates, roll_rhos, color="#D84315", linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Cross-Correlation (rho)")
    ax2.set_xlabel("Date")

    colours = {"COVID-19": "#C62828", "Ukraine": "#6A1B9A"}
    for bname, bdate in BREAKS.items():
        col = colours[bname]
        for ax in (ax1, ax2):
            ax.axvline(pd.Timestamp(bdate), color=col,
                       linestyle="--", linewidth=1.0, label=bname)
    ax1.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "fig_c2c_structural_break_steel.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"fig_c2c_structural_break_steel.pdf saved.")

    # ── Rolling lag figure — all 5 pairs ──────────────────────────────────
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(10, 2.2 * len(PAIRS)),
                              sharex=True)
    for idx, (us_col, gr_col, plabel) in enumerate(PAIRS):
        if us_col not in df.columns or gr_col not in df.columns:
            continue
        r_dates, r_lags = [], []
        for i in range(ROLL_WINDOW, len(df)):
            y_w = df[gr_col].iloc[i - ROLL_WINDOW:i].values
            x_w = df[us_col].iloc[i - ROLL_WINDOW:i].values
            l_w, _ = best_lag(y_w, x_w)
            r_dates.append(df.index[i])
            r_lags.append(l_w)
        ax = axes[idx]
        ax.plot(r_dates, r_lags, color="#1565C0", linewidth=1)
        ax.set_ylabel(f"{plabel}\nlag (M)", fontsize=8)
        ax.set_yticks(range(0, MAX_LAG + 1))
        for bname, bdate in BREAKS.items():
            ax.axvline(pd.Timestamp(bdate), color=colours[bname],
                       linestyle="--", linewidth=0.8)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c2c_rolling_lag_all.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c2c_rolling_lag_all.pdf saved.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\nStructural Break Summary:")
    cols = ["Pair", "Break", "Chow_p", "Stars",
            "Lag_pre", "Lag_post", "Lag_change", "Rho_change"]
    print(out_df[cols].to_string(index=False))
    print(f"\nSaved: c2c_structural_breaks.csv")


if __name__ == "__main__":
    main()
