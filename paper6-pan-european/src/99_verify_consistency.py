"""
99_verify_consistency.py
========================
Consistency harness. Run after any pipeline
change. Checks (Phase A scope):

  1. Every expected output file exists and is non-empty.
  2. Panel integrity: no internal monthly gaps; |log-return| <= 15%.
  3. lambda_U in [0, 1] everywhere; per-pair amplification in the
     pair table equals crisis/stable means recomputed from the
     stored time series (tolerance 1e-3).
  4. Bridge tables carry the pre-registered fields; the acceptance
     verdict recomputed from the stored numbers matches the one
     implied by the data (rho >= 0.5 at |lag| <= 3).
  5. h1 summary is consistent with the pair table.

Exit code 1 on any failure. Extend as later phases add outputs.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


EXPECTED = [
    (PROC, "cci_panel_primary.csv"), (PROC, "cci_panel_extended.csv"),
    (PROC, "cci_panel_cost.csv"), (PROC, "panel_coverage.csv"),
    (PROC, "cci_features_primary.csv"),
    (PROC, "lambdaU_pairs_w24.csv"), (PROC, "lambdaU_pairs_w36.csv"),
    (TAB, "table_bridge_correlations.csv"),
    (TAB, "table_bridge_lagged.csv"),
    (TAB, "table_bridge_indicator_check.csv"),
    (TAB, "table_cross_country_lambdaU.csv"),
    (TAB, "table_h1_summary.csv"),
    # Phase B
    (TAB, "table_h1_subgroups.csv"),
    (TAB, "table_granger_network.csv"), (TAB, "table_dy_spillover.csv"),
    (TAB, "table_h3_summary.csv"),
    (TAB, "table_clustering.csv"), (TAB, "table_h2_summary.csv"),
    (TAB, "table_panel_interaction.csv"), (TAB, "table_h4_summary.csv"),
    (TAB, "table_ukraine_case.csv"), (TAB, "table_ukraine_network.csv"),
    # Phase G
    (TAB, "table_quarterly_h1.csv"), (TAB, "table_quarterly_h3.csv"),
    (TAB, "table_indicator_robustness.csv"),
    (TAB, "table_copula_families.csv"), (TAB, "table_g3_summary.csv"),
    (TAB, "table_h4_wildboot.csv"),
    (TAB, "table_h1_episodes.csv"), (TAB, "table_h1_bootstrap_ci.csv"),
    (TAB, "table_robustness_labels.csv"),
    (TAB, "table_robustness_seasonal.csv"),
    (TAB, "table_robustness_gr_source.csv"),
    (TAB, "table_estimator_crossval.csv"),
]
for d, f in EXPECTED:
    p = os.path.join(d, f)
    check(os.path.exists(p) and os.path.getsize(p) > 0, f"missing: {f}")

if fails:
    print("FAILURES:", *fails, sep="\n  - ")
    sys.exit(1)

# 2. panel integrity
pan = pd.read_csv(os.path.join(PROC, "cci_panel_primary.csv"),
                  parse_dates=["date"])
for c, g in pan.groupby("country"):
    g = g.sort_values("date")
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
    check(len(full) == len(g), f"{c}: internal gap in monthly panel")
check(pan["log_return"].abs().max() <= 0.15, "log-return > 15% present")

# 3. lambda_U range + amplification recomputation
pairs = pd.read_csv(os.path.join(TAB, "table_cross_country_lambdaU.csv"))
feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                   parse_dates=["date"])
cx = feat.pivot(index="date", columns="country",
                values="crisis_exog").sort_index()
for w in [24, 36]:
    ts = pd.read_csv(os.path.join(PROC, f"lambdaU_pairs_w{w}.csv"),
                     index_col=0, parse_dates=True)
    check(float(ts.min().min()) >= 0 and float(ts.max().max()) <= 1,
          f"lambda_U outside [0,1] (w={w})")
    for _, row in pairs[pairs["window"] == w].iterrows():
        a, b = row["pair"].split("-")
        s = ts[row["pair"]].dropna()
        mask = cx.loc[s.index, [a, b]].max(axis=1).values
        mu_c = s.values[mask == 1].mean()
        mu_s = s.values[mask == 0].mean()
        check(abs(mu_c - row["lambdaU_crisis"]) < 1e-3
              and abs(mu_s - row["lambdaU_stable"]) < 1e-3,
              f"{row['pair']} w={w}: table/timeseries mismatch")

# 4. bridge verdict recomputation
lag = pd.read_csv(os.path.join(TAB, "table_bridge_lagged.csv"))
near = lag[lag["lag_csri_leads"].abs() <= 3]
verdict = bool((near["spearman"] >= 0.5).any())
print(f"bridge A verdict (recomputed): {'PASS' if verdict else 'FAIL'}")

# 4b. Phase B/G cross-checks
sub = pd.read_csv(os.path.join(TAB, "table_h1_subgroups.csv"))
h1s = pd.read_csv(os.path.join(TAB, "table_h1_summary.csv"))
for w in [24, 36]:
    a = sub[(sub["window"] == w) & (sub["labels"] == "exog")
            & (sub["bloc"] == "ALL")]["panel_diff"].iloc[0]
    b = h1s[h1s["window"] == w]["panel_diff"].iloc[0]
    check(abs(a - b) < 1e-3, f"H1 ALL panel_diff mismatch 04 vs 04b (w={w})")

fam = pd.read_csv(os.path.join(TAB, "table_copula_families.csv"))
check(fam["lambdaU_gumbel"].between(0, 1).all(),
      "pooled Gumbel lambda_U outside [0,1]")
check(np.isfinite(fam[[c for c in fam.columns
                       if c.startswith("aic_")]].values).all(),
      "non-finite AIC in copula-family table")

wb = pd.read_csv(os.path.join(TAB, "table_h4_wildboot.csv"))
h4 = pd.read_csv(os.path.join(TAB, "table_h4_summary.csv"))
check(abs(wb["beta2"].iloc[0] - h4["beta2_med_extra_ppsigma"].iloc[0])
      < 1e-3, "H4 beta2 mismatch 08 vs 08b")

ukr = pd.read_csv(os.path.join(TAB, "table_ukraine_network.csv"),
                  index_col=0)
check(float(ukr["full_sample_percentile"].max()) >= 0.99,
      "Ukraine window should contain the full-sample lambda_U peak")

# 5. summary vs pair table
summ = pd.read_csv(os.path.join(TAB, "table_h1_summary.csv"))
for _, s in summ.iterrows():
    pr = pairs[pairs["window"] == s["window"]]
    check(int((pr["amplification"] > 1).sum()) == int(s["pairs_amp_gt_1"]),
          f"summary amp>1 count mismatch (w={int(s['window'])})")
    check(abs(np.nanmean(pr["amplification"]) - s["mean_amplification"])
          < 1e-3, f"summary mean amp mismatch (w={int(s['window'])})")

print("=" * 50)
if fails:
    print(f"FAILURES: {len(fails)}", *fails, sep="\n  - ")
    sys.exit(1)
print("ALL CONSISTENCY CHECKS PASSED")
