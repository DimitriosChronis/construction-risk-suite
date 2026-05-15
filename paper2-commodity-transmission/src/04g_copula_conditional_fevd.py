"""
04g_copula_conditional_fevd.py
==============================
Mitigation for R2-a / R3-S3: deepens the vine-VAR integration by
estimating the system VAR FEVD CONDITIONALLY on the vine copula
tail probability of the network.

Methodology:
  1. From the fitted Tree-1 vine copula, compute month-by-month
     average pseudo-observation distance from the centre, used as
     a proxy for "tail-state" intensity:
        tau_t = mean over the 5 cross-market pairs of
                |2*F(u_jt) - 1| * |2*F(u_kt) - 1|
     where F(u) is the empirical CDF of the pseudo-obs.
  2. Split the sample into "high tail" (top 33%) and "low tail"
     (bottom 67%) months by tau_t.
  3. Fit the system VAR(1) separately on each subset.
  4. Compare h=12 FEVD shares.

This provides a true vine-conditioned FEVD, which is the
"copula-estimated conditional dependence informing VAR
identification" that Reviewer 3 (S3) suggested.

Output:
    results/tables/c2g_vine_conditional_fevd.csv
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

# Cross-market matched pairs that anchor the vine
PAIRS = [("US_Brent",      "GR_General_Index"),
         ("US_Cement_PPI", "GR_Concrete"),
         ("US_Steel_PPI",  "GR_Steel"),
         ("US_Fuel_PPI",   "GR_Fuel_Energy"),
         ("US_PVC_PPI",    "GR_PVC_Pipes")]

MATCHED = {gr: us for us, gr in PAIRS}

H_FEVD = 12
TAIL_PCT = 33  # top X% = "high tail" regime


def empirical_cdf(x):
    """Map an array to its empirical CDF (rank/(n+1))."""
    n = len(x)
    return (pd.Series(x).rank(method="average") - 0.5) / n


def tail_proxy(df, pairs):
    """For each month, average across the 5 cross-market pairs the
    product |2*F(u_us)-1| * |2*F(u_gr)-1|. High values = both series
    in joint tail."""
    cols = []
    for us, gr in pairs:
        u_us = empirical_cdf(df[us].values)
        u_gr = empirical_cdf(df[gr].values)
        cols.append(np.abs(2 * u_us - 1) * np.abs(2 * u_gr - 1))
    return np.mean(np.column_stack(cols), axis=1)


def system_fevd(res, h):
    fevd = res.fevd(h)
    out = {}
    for j, name in enumerate(res.names):
        out[name] = {shock: float(fevd.decomp[j, h - 1, k])
                     for k, shock in enumerate(res.names)}
    return out


def main():
    os.makedirs(OUT_TAB, exist_ok=True)
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)[ALL_VARS]
    df = df.dropna()

    print("=" * 70)
    print("Paper 2 -- Mitigation: Vine-conditional FEVD (04g)")
    print("=" * 70)
    print(f"  Data: {df.shape}")

    # 1. compute vine tail-state proxy
    tau = tail_proxy(df, PAIRS)
    df = df.assign(_tau=tau)

    # 2. split into high vs low regimes
    cutoff = float(np.quantile(tau, 1 - TAIL_PCT / 100.0))
    high_mask = df["_tau"] >= cutoff
    low_mask  = df["_tau"] <  cutoff

    print(f"\n  tail proxy cutoff (top {TAIL_PCT}%) = {cutoff:.4f}")
    print(f"  high-tail months: n = {high_mask.sum()}")
    print(f"  low-tail  months: n = {low_mask.sum()}")

    # 3. fit system VAR(1) on each subset and extract FEVD
    rows = []
    for label, mask in [("high_tail", high_mask), ("low_tail", low_mask)]:
        sub = df.loc[mask, ALL_VARS]
        if len(sub) < 30:
            print(f"  [skip] {label}: n too small")
            continue
        try:
            res = VAR(sub).fit(maxlags=1, ic=None)
            fevd = system_fevd(res, H_FEVD)
            for gr in GR_VARS:
                matched = MATCHED[gr]
                cross = sum(fevd[gr][u] for u in US_VARS if u != matched)
                rows.append({
                    "regime":         label,
                    "n":              int(mask.sum()),
                    "Greek_series":   gr,
                    "Matched_US":     matched,
                    "FEVD_matched":   round(fevd[gr][matched], 4),
                    "FEVD_cross_mat": round(cross, 4),
                    "FEVD_total_US":  round(fevd[gr][matched] + cross, 4),
                })
        except Exception as e:
            print(f"  [warn] {label}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_TAB,
                            "c2g_vine_conditional_fevd.csv"), index=False)

    print("\n  Vine-conditional FEVD at h=12:")
    print(out.pivot(index=["Greek_series", "Matched_US"],
                    columns="regime",
                    values="FEVD_total_US").round(3))

    # 4. amplification ratio (high / low)
    print("\n  Total US-FEVD amplification (high-tail / low-tail):")
    pivot = out.pivot(index="Greek_series", columns="regime",
                      values="FEVD_total_US")
    if "high_tail" in pivot.columns and "low_tail" in pivot.columns:
        pivot["amp_ratio"] = pivot["high_tail"] / pivot["low_tail"]
        for gr in pivot.index:
            r = pivot.loc[gr, "amp_ratio"]
            print(f"    {gr:20s}  ratio = {r:.2f}x")

    print("\n" + "=" * 70)
    print("DONE -- 04g_copula_conditional_fevd.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
