"""
03b_bridge_validation.py
========================
H1b bridge: does the Greek MACRO construction-cost signal track the
MICRODATA-derived systemic-risk signal, licensing macro indices as
the cross-country unit of analysis?

Bridge A (pre-registered acceptance: Spearman >= 0.5 at lag 0, OR a
clear lead-lag peak within +/-3 months):
    CSRI_GR (micro; from the public Diavgeia pipeline, 2002-2024)
    vs GR macro CCI 6M volatility (sigma6).

Bridge B (GR indicator cross-check): ELSTAT monthly cost index
aggregated to quarters vs Eurostat EL quarterly COST and PRC_PRR
(log-return correlations) -- answers the "GR uses a national cost
index while others use Eurostat producer prices" concern.

Inputs : data/processed/cci_features_primary.csv
         ../paper5-portfolio-contagion/data/processed/csri_monthly.csv
         data/raw/eurostat_cci_quarterly.csv, data/raw/cci_GR_elstat.csv
Outputs: results/tables/table_bridge_correlations.csv
         results/tables/table_bridge_lagged.csv
         results/tables/table_bridge_indicator_check.csv
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")
CSRI_PATH = os.path.join(SCRIPT_DIR, "..", "..",
                         "paper5-portfolio-contagion", "data",
                         "processed", "csri_monthly.csv")

ACCEPT_RHO = 0.5
MAX_LAG = 6


def main():
    os.makedirs(TAB, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- H1b bridge validation")
    print("=" * 70)

    # ---------------- Bridge A: CSRI (micro) vs GR sigma6 (macro) ----
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    gr = feat[feat["country"] == "GR"].set_index("date")
    csri = pd.read_csv(CSRI_PATH, index_col=0, parse_dates=True)["CSRI"]

    j = pd.DataFrame({"csri": csri, "sigma6": gr["sigma6"]}).dropna()
    print(f"[A] overlap: {len(j)} months "
          f"[{j.index.min():%Y-%m} -> {j.index.max():%Y-%m}]")

    rows = []
    for name, fn in [("pearson", stats.pearsonr),
                     ("spearman", stats.spearmanr),
                     ("kendall", stats.kendalltau)]:
        r, p = fn(j["csri"], j["sigma6"])
        rows.append({"bridge": "A_csri_vs_sigma6", "metric": name,
                     "estimate": round(float(r), 4),
                     "p_value": float(f"{p:.3e}"), "n": len(j)})
        print(f"  {name:9s} r = {r:+.3f}  (p = {p:.2e})")
    pd.DataFrame(rows).to_csv(
        os.path.join(TAB, "table_bridge_correlations.csv"), index=False)

    lag_rows = []
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        s = j["sigma6"].shift(-lag)   # lag>0: CSRI leads sigma6
        m = pd.DataFrame({"a": j["csri"], "b": s}).dropna()
        r, p = stats.spearmanr(m["a"], m["b"])
        lag_rows.append({"lag_csri_leads": lag,
                         "spearman": round(float(r), 4),
                         "p_value": float(f"{p:.3e}"), "n": len(m)})
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(os.path.join(TAB, "table_bridge_lagged.csv"), index=False)
    best = lag_df.loc[lag_df["spearman"].idxmax()]
    print(f"  best lag: CSRI leads sigma6 by {int(best['lag_csri_leads'])}M "
          f"(rho = {best['spearman']:+.3f})")

    rho0 = lag_df.loc[lag_df["lag_csri_leads"] == 0, "spearman"].iloc[0]
    ok = (rho0 >= ACCEPT_RHO) or (
        best["spearman"] >= ACCEPT_RHO
        and abs(best["lag_csri_leads"]) <= 3)
    print(f"  PRE-REGISTERED ACCEPTANCE (rho>= {ACCEPT_RHO} at |lag|<=3): "
          f"{'PASS' if ok else 'FAIL'}")

    # ---------------- Bridge B: ELSTAT vs Eurostat EL (quarterly) ----
    print("\n[B] GR indicator cross-check (quarterly log-returns):")
    el = pd.read_csv(os.path.join(RAW, "eurostat_cci_quarterly.csv"),
                     parse_dates=["date"])
    el = el[el["country"] == "EL"]
    grm = pd.read_csv(os.path.join(RAW, "cci_GR_elstat.csv"),
                      parse_dates=["date"]).set_index("date")["level"]
    grq = np.log(grm.resample("QE").mean()).diff().dropna()
    grq.index = grq.index.to_period("Q")

    rows_b = []
    for ind, g in el.groupby("indicator"):
        s = g.set_index("date")["level"].sort_index()
        sq = np.log(s).diff().dropna()
        sq.index = pd.PeriodIndex(sq.index, freq="Q")
        m = pd.DataFrame({"elstat": grq, "eurostat": sq}).dropna()
        rp, pp = stats.pearsonr(m["elstat"], m["eurostat"])
        rs, ps = stats.spearmanr(m["elstat"], m["eurostat"])
        rows_b.append({"indicator": ind, "n_quarters": len(m),
                       "pearson": round(float(rp), 4),
                       "pearson_p": float(f"{pp:.3e}"),
                       "spearman": round(float(rs), 4),
                       "spearman_p": float(f"{ps:.3e}")})
        print(f"  ELSTAT vs EL {ind:8s}: pearson {rp:+.3f} (p={pp:.1e})  "
              f"spearman {rs:+.3f} (p={ps:.1e})  n={len(m)}")
    pd.DataFrame(rows_b).to_csv(
        os.path.join(TAB, "table_bridge_indicator_check.csv"), index=False)

    print("\nDONE -- 03b_bridge_validation.py")


if __name__ == "__main__":
    main()
