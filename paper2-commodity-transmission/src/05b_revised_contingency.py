"""
05b_revised_contingency.py
==========================
Revision response to R1-M2, R2-b, R3-§4 (EUR contingency
inconsistent with copula evidence and uses inappropriate sqrt(H)
scaling for serially correlated returns).

Two corrections:

  (A) FEVD basis: replace bivariate FEVD with SYSTEM FEVD computed
      in 04d_system_varx.py. The system FEVD attributes Greek-series
      variance jointly across all five US shocks and removes
      double-counting between the matched and cross-material channels.

  (B) Time scaling: replace sigma * sqrt(H) with a bootstrap of
      H-month CUMULATIVE log returns drawn from the empirical
      panel. Following moving-block resampling (block length L = 6
      months) the bootstrap preserves serial correlation and the
      H-period distributional shape that sqrt-H scaling assumes
      away. Conditional VaR (= P5 contingency) is then read from
      the empirical 5th percentile of the H-month bootstrap returns.

Inputs:
    data/processed/aligned_log_returns.csv
    results/tables/c2d_system_fevd.csv                (from 04d)
    results/tables/c2d_bivariate_vs_system_fevd.csv   (from 04d)
Outputs:
    results/tables/c2_5b_revised_contingency.csv
    results/tables/c2_5b_scaling_comparison.csv
    results/tables/c2_5b_portfolio_totals.csv

Consistency notes (final-version fix):
  * The H-month bootstrap VaR is drawn ONCE per Greek series and the
    same draw feeds both the scaling-comparison table and the
    contingency table, so that (D)/(B) equals the reported
    bootstrap-to-sqrt(H) ratio exactly, material by material.
  * Specification (C) uses the BIVARIATE FEVD share (from 04d's
    bivariate-vs-system comparison), not the system share, so that
    (C) and (D) differ as their definitions require.
  * All totals are computed as sums of the per-material entries.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                          "aligned_log_returns.csv")
TAB_DIR    = os.path.join(SCRIPT_DIR, "..", "results", "tables")
SYS_FEVD   = os.path.join(TAB_DIR, "c2d_system_fevd.csv")
BIV_FEVD   = os.path.join(TAB_DIR, "c2d_bivariate_vs_system_fevd.csv")

# Project parameters (from v1 manuscript Section 4.7)
PROJECT_BUDGET_EUR = 2_300_000          # representative project
HORIZON_MONTHS     = 24                 # construction lifecycle
N_BOOT             = 5000               # bootstrap replicates
BLOCK_LEN          = 6                  # moving-block length (months)
ALPHA              = 0.05               # 5% tail (P95 contingency)
SEED               = 42

# Material weights from the v1 manuscript (Table 4)
WEIGHTS = {
    "GR_Concrete":      0.30,
    "GR_Steel":         0.25,
    "GR_Fuel_Energy":   0.20,
    "GR_PVC_Pipes":     0.10,
    "GR_General_Index": 0.15,
}

US_VARS = ["US_Brent", "US_Steel_PPI", "US_Cement_PPI",
           "US_Fuel_PPI", "US_PVC_PPI"]
GR_VARS = list(WEIGHTS.keys())

MATCHED = {
    "GR_Steel":         "US_Steel_PPI",
    "GR_Concrete":      "US_Cement_PPI",
    "GR_General_Index": "US_Brent",
    "GR_Fuel_Energy":   "US_Fuel_PPI",
    "GR_PVC_Pipes":     "US_PVC_PPI",
}

# v1 contingency values (h=24, sqrt-H, bivariate FEVD), matching the
# original published Table 4 of the v1 manuscript:
#   Total (4 materials) = 45,174
#   Steel:    22,779   (FEVD 22.2%)
#   Fuel:     12,254   (FEVD  2.9%)
#   Cement:    5,102   (FEVD  7.3%)
#   PVC:       5,039   (FEVD  4.9%)
#   General/Brent reported separately as 25,388 (full-project basis,
#   not commensurable with the per-material rows). We exclude it
#   here for an apples-to-apples comparison with the revision.
V1_CONTINGENCY = {
    "GR_Steel":         22_779,
    "GR_Fuel_Energy":   12_254,
    "GR_Concrete":       5_102,
    "GR_PVC_Pipes":      5_039,
    "GR_General_Index":      0,   # excluded; reported separately in v1
}
V1_GENERAL_BRENT_REPORTED = 25_388  # for narrative reconciliation only

# Material-cost share assumed in v1 (Section 4.7)
MATERIAL_SHARE = 0.55      # fraction of total project cost that is
                           # exposed to commodity shocks


def moving_block_bootstrap_paths(returns, n_paths, h, block_len, rng):
    """Generate n_paths H-period log-return paths via moving-block
    resampling. Returns the H-period CUMULATIVE log-return for each
    path, as an array of shape (n_paths,).
    """
    r = returns.dropna().values
    n = len(r)
    n_blocks = int(np.ceil(h / block_len))
    cum = np.empty(n_paths)
    for i in range(n_paths):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        path = np.concatenate([r[s:s + block_len] for s in starts])[:h]
        cum[i] = path.sum()
    return cum


def main():
    os.makedirs(TAB_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df[US_VARS + GR_VARS].dropna()
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 2 -- Revision: Revised contingency (system FEVD + bootstrap)")
    print("=" * 70)
    print(f"  Data: {df.shape}")
    print(f"  Project budget: EUR {PROJECT_BUDGET_EUR:,}")
    print(f"  Horizon: {HORIZON_MONTHS} months  Bootstrap: B={N_BOOT}, L={BLOCK_LEN}")

    # -------------------------------------------------------------------------
    # 1. LOAD SYSTEM FEVD FROM 04d
    # -------------------------------------------------------------------------
    if not os.path.exists(SYS_FEVD):
        raise FileNotFoundError(
            f"System FEVD not found at {SYS_FEVD}. "
            "Run 04d_system_varx.py first.")
    fevd_long = pd.read_csv(SYS_FEVD)
    fevd_wide = fevd_long.pivot(index="response", columns="shock",
                                 values="share")
    print("\n[1] System FEVD loaded:")
    print(fevd_wide.round(3))

    # Bivariate FEVD shares (matched channel) from 04d, used by spec C
    if not os.path.exists(BIV_FEVD):
        raise FileNotFoundError(
            f"Bivariate FEVD comparison not found at {BIV_FEVD}. "
            "Run 04d_system_varx.py first.")
    biv_df = pd.read_csv(BIV_FEVD).set_index("Greek_series")
    biv_share = biv_df["Bivariate_FEVD"].astype(float).to_dict()
    print("\n[1b] Bivariate FEVD (matched channel) loaded:",
          {k: round(v, 3) for k, v in biv_share.items()})

    # -------------------------------------------------------------------------
    # 2. SCALING COMPARISON: sqrt-H vs bootstrap H-period
    #    (ONE bootstrap draw per series; reused in step 3 so that the
    #     contingency ratios (D)/(B) equal ratio_boot_to_sqrtH exactly)
    # -------------------------------------------------------------------------
    print("\n[2] Scaling comparison: sqrt-H vs H-period block bootstrap")
    scale_rows = []
    var_sqrtH_by = {}
    var_boot_by = {}
    for gr in GR_VARS:
        sigma_1m = float(df[gr].std())
        # sqrt-H tail (one-sided)
        var_sqrtH = 1.645 * sigma_1m * np.sqrt(HORIZON_MONTHS)
        # bootstrap H-period left tail (P5)
        cum_paths = moving_block_bootstrap_paths(
            df[gr], N_BOOT, HORIZON_MONTHS, BLOCK_LEN, rng)
        var_boot = -float(np.quantile(cum_paths, ALPHA))
        var_sqrtH_by[gr] = var_sqrtH
        var_boot_by[gr] = var_boot
        scale_rows.append({
            "Greek_series":   gr,
            "sigma_1m":       round(sigma_1m, 5),
            "VaR_sqrtH":      round(var_sqrtH, 4),
            "VaR_bootstrap":  round(var_boot, 4),
            "ratio_boot_to_sqrtH": round(var_boot / var_sqrtH, 3),
        })
        print(f"  {gr:20s}  sigma={sigma_1m:.4f}  "
              f"sqrtH-VaR={var_sqrtH:.3f}  bootVaR={var_boot:.3f}  "
              f"ratio={var_boot / var_sqrtH:+.3f}")
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(os.path.join(TAB_DIR, "c2_5b_scaling_comparison.csv"),
                    index=False)

    # -------------------------------------------------------------------------
    # 3. CONTINGENCY UNDER 4 SPECIFICATIONS
    # -------------------------------------------------------------------------
    # Spec A : v1 published (bivariate FEVD + sqrt-H)   -- textbook benchmark
    # Spec B : system FEVD + sqrt-H        (R1-M2 fix on FEVD)
    # Spec C : bivariate FEVD + bootstrap  (R3-S4 fix on scaling)
    # Spec D : system FEVD + bootstrap     (FULL revision -- preferred)
    # Contingency = exposure x VaR x sqrt(FEVD share)
    # (sqrt(FEVD) ~ standard-deviation share; consistent with v1)
    print("\n[3] Contingency under 4 specifications:")
    cont_rows = []
    for gr in GR_VARS:
        matched_us = MATCHED[gr]
        sys_share = float(fevd_wide.loc[gr, matched_us])
        sys_cross = float(fevd_wide.loc[gr, [u for u in US_VARS
                                             if u != matched_us]].sum())
        sys_total = sys_share + sys_cross
        b_share   = float(biv_share[gr])
        v1_eur    = V1_CONTINGENCY.get(gr, np.nan)

        w       = WEIGHTS[gr]
        exp_eur = PROJECT_BUDGET_EUR * MATERIAL_SHARE * w
        var_sqrtH = var_sqrtH_by[gr]      # same draw as Table (scaling)
        var_boot  = var_boot_by[gr]

        eur_B_match = exp_eur * var_sqrtH * np.sqrt(sys_share)
        eur_B_total = exp_eur * var_sqrtH * np.sqrt(sys_total)
        eur_C_match = exp_eur * var_boot  * np.sqrt(b_share)
        eur_D_match = exp_eur * var_boot  * np.sqrt(sys_share)
        eur_D_total = exp_eur * var_boot  * np.sqrt(sys_total)

        cont_rows.append({
            "Greek_series":       gr,
            "Matched_US":         matched_us,
            "FEVD_bivariate_match": round(b_share, 4),
            "FEVD_system_match":  round(sys_share, 4),
            "FEVD_system_cross":  round(sys_cross, 4),
            "EUR_A_v1_published": int(v1_eur) if not np.isnan(v1_eur) else np.nan,
            "EUR_B_sysFEVD_sqrtH":      int(round(eur_B_match)),
            "EUR_B_sysFEVD_total_sqrtH":int(round(eur_B_total)),
            "EUR_C_bivFEVD_boot":       int(round(eur_C_match)),
            "EUR_D_sysFEVD_boot":       int(round(eur_D_match)),
            "EUR_D_total_sysFEVD_boot": int(round(eur_D_total)),
        })

    cont_df = pd.DataFrame(cont_rows)
    cont_df.to_csv(os.path.join(TAB_DIR, "c2_5b_revised_contingency.csv"),
                   index=False)
    print(cont_df[["Greek_series", "EUR_A_v1_published",
                   "EUR_B_sysFEVD_sqrtH", "EUR_C_bivFEVD_boot",
                   "EUR_D_sysFEVD_boot", "EUR_D_total_sysFEVD_boot"]]
          .to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. PORTFOLIO TOTALS  (all totals = sums of per-material entries)
    # -------------------------------------------------------------------------
    print("\n[4] Portfolio totals (EUR per representative EUR 2.3M project):")
    four = cont_df["Greek_series"] != "GR_General_Index"
    totals = {
        "A_v1_published_4mat":     cont_df.loc[four, "EUR_A_v1_published"].sum(),
        "B_sysFEVD_sqrtH_4mat":    cont_df.loc[four, "EUR_B_sysFEVD_sqrtH"].sum(),
        "B_sysFEVD_sqrtH_5ch":     cont_df["EUR_B_sysFEVD_sqrtH"].sum(),
        "B_sysFEVD_total_sqrtH":   cont_df["EUR_B_sysFEVD_total_sqrtH"].sum(),
        "C_bivFEVD_boot_4mat":     cont_df.loc[four, "EUR_C_bivFEVD_boot"].sum(),
        "C_bivFEVD_boot_5ch":      cont_df["EUR_C_bivFEVD_boot"].sum(),
        "D_sysFEVD_boot_4mat":     cont_df.loc[four, "EUR_D_sysFEVD_boot"].sum(),
        "D_sysFEVD_boot_5ch":      cont_df["EUR_D_sysFEVD_boot"].sum(),
        "D_total_sysFEVD_boot":    cont_df["EUR_D_total_sysFEVD_boot"].sum(),
    }
    for k, v in totals.items():
        v_pct = 100.0 * v / PROJECT_BUDGET_EUR if v == v else np.nan
        print(f"  {k:30s}  EUR {int(v):>9,}  ({v_pct:5.2f}% of budget)")

    pd.Series(totals, name="EUR_total").to_csv(
        os.path.join(TAB_DIR, "c2_5b_portfolio_totals.csv"))

    # Consistency check: (D)/(B) must equal the scaling ratio per series
    chk = cont_df.set_index("Greek_series")
    for gr in GR_VARS:
        r_tab = float(scale_df.set_index("Greek_series").loc[gr, "ratio_boot_to_sqrtH"])
        r_con = chk.loc[gr, "EUR_D_sysFEVD_boot"] / chk.loc[gr, "EUR_B_sysFEVD_sqrtH"]
        print(f"  check {gr:18s} D/B={r_con:.3f}  table ratio={r_tab:.3f}")

    print("\n" + "=" * 70)
    print("DONE -- 05b_revised_contingency.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
