"""
04f_regime_fevd_bootstrap.py
============================
Mitigation for Reviewer concern on small post-COVID sample (n=58).

Bootstrap confidence intervals for the regime-conditional system
FEVD reported in 04d (Table c2d_regime_conditional_fevd.csv).

Methodology: B=500 moving-block bootstrap replicates (block length
12 months) of the post-COVID sub-sample. For each replicate, refit
the system VAR(1) and compute h=12 FEVD shares for matched
US -> Greek pairs. Report 95% bootstrap CIs.

Output:
    results/tables/c2d_regime_fevd_bootstrap_ci.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                          "aligned_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

US_VARS = ["US_Brent", "US_Steel_PPI", "US_Cement_PPI",
           "US_Fuel_PPI", "US_PVC_PPI"]
GR_VARS = ["GR_General_Index", "GR_Concrete", "GR_Steel",
           "GR_Fuel_Energy", "GR_PVC_Pipes"]
ALL_VARS = US_VARS + GR_VARS
MATCHED = {
    "GR_Steel":         "US_Steel_PPI",
    "GR_Concrete":      "US_Cement_PPI",
    "GR_General_Index": "US_Brent",
    "GR_Fuel_Energy":   "US_Fuel_PPI",
    "GR_PVC_Pipes":     "US_PVC_PPI",
}

H_FEVD      = 12
N_BOOT      = 500
BLOCK_LEN   = 12
SEED        = 42
COVID_BREAK = "2020-03-01"


def block_bootstrap_indices(n, block_len, rng):
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def system_fevd_at_h(res, h):
    fevd = res.fevd(h)
    out = {}
    for j, name in enumerate(res.names):
        out[name] = {shock: float(fevd.decomp[j, h - 1, k])
                     for k, shock in enumerate(res.names)}
    return out


def main():
    os.makedirs(OUT_TAB, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)[ALL_VARS]
    df = df.dropna()

    print("=" * 70)
    print("Paper 2 -- Mitigation: Regime FEVD bootstrap CI (04f)")
    print("=" * 70)

    for label, mask in [("pre_COVID",  df.index <  pd.Timestamp(COVID_BREAK)),
                         ("post_COVID", df.index >= pd.Timestamp(COVID_BREAK))]:
        sub = df[mask].values
        n = len(sub)
        print(f"\n[{label}]  n = {n}")
        if n < 30:
            print("  too small to bootstrap reliably; skipping.")
            continue

        boot_fevd = {gr: [] for gr in GR_VARS}
        good = 0
        for b in range(N_BOOT):
            idx = block_bootstrap_indices(n, BLOCK_LEN, rng)
            try:
                Z = pd.DataFrame(sub[idx], columns=ALL_VARS)
                res = VAR(Z).fit(maxlags=1, ic=None)
                fevd = system_fevd_at_h(res, H_FEVD)
                for gr in GR_VARS:
                    boot_fevd[gr].append(fevd[gr][MATCHED[gr]])
                good += 1
            except Exception:
                continue
        print(f"  {good}/{N_BOOT} successful replicates")

        rows = []
        for gr in GR_VARS:
            arr = np.asarray(boot_fevd[gr])
            rows.append({
                "regime":       label,
                "Greek_series": gr,
                "Matched_US":   MATCHED[gr],
                "mean":     round(float(arr.mean()),       4),
                "median":   round(float(np.median(arr)),   4),
                "ci_low":   round(float(np.quantile(arr, 0.025)), 4),
                "ci_high":  round(float(np.quantile(arr, 0.975)), 4),
                "std":      round(float(arr.std()),        4),
                "n_boot":   good,
            })
        out_df = pd.DataFrame(rows)
        path = os.path.join(OUT_TAB,
                            f"c2d_regime_fevd_bootstrap_ci_{label}.csv")
        out_df.to_csv(path, index=False)
        print(out_df[["Greek_series", "mean", "ci_low", "ci_high", "std"]])

    print("\n" + "=" * 70)
    print("DONE -- 04f_regime_fevd_bootstrap.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
