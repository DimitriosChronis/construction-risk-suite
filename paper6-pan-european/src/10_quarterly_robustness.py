"""
10_quarterly_robustness.py
==========================
G1: the quarterly robustness layer -- brings DE and FR (which have
no post-2024 monthly source) into the analysis and re-runs the two
headline tests at quarterly frequency for the full 10-country set.

Panel: primary 8 aggregated to quarterly + DE, FR (Eurostat
sts_copi_q, PRC_PRR). Tests: (i) H1-style panel amplification of
rolling 12-quarter pair-Gumbel lambda_U under the exogenous crisis
windows; (ii) H3-style Granger link share (max lag 2 quarters).

Inputs : data/processed/cci_panel_primary.csv,
         data/raw/eurostat_cci_quarterly.csv
Outputs: results/tables/table_quarterly_h1.csv
         results/tables/table_quarterly_h3.csv
"""

import os
import warnings
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

W_Q = 12          # 12 quarters = 36 months
B_PERM = 1000
SEED = 42
MED = {"GR", "ES", "IT", "PT"}
EXOG_Q = [("2008Q3", "2009Q4"), ("2011Q3", "2012Q4"),
          ("2020Q1", "2020Q4"), ("2022Q1", "2022Q4")]


def lam_from_tau(tau):
    if tau is None or np.isnan(tau) or tau <= 0:
        return 0.0
    theta = 1.0 / (1.0 - min(tau, 0.99))
    return 2.0 - 2.0 ** (1.0 / theta)


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- G1 quarterly robustness (with DE, FR)")
    print("=" * 70)

    mon = pd.read_csv(os.path.join(PROC, "cci_panel_primary.csv"),
                      parse_dates=["date"])
    frames = {}
    for c, g in mon.groupby("country"):
        lvl = g.set_index("date")["level"].resample("QE").mean()
        frames[c] = np.log(lvl).diff()
    q = pd.read_csv(os.path.join(RAW, "eurostat_cci_quarterly.csv"),
                    parse_dates=["date"])
    for c in ["DE", "FR"]:
        s = q[(q["country"] == c) & (q["indicator"] == "PRC_PRR")]
        lvl = s.set_index("date")["level"].resample("QE").last()
        frames[c] = np.log(lvl).diff()
    wide = pd.DataFrame(frames).dropna(how="all")
    wide.index = wide.index.to_period("Q")
    countries = sorted(wide.columns)
    print(f"quarterly panel: {len(countries)} countries, "
          f"{wide.index.min()} -> {wide.index.max()}")

    mask_full = pd.Series(0, index=wide.index)
    for a, b in EXOG_Q:
        mask_full.loc[pd.Period(a):pd.Period(b)] = 1

    # ---- H1-style amplification ----
    ts = {}
    for a, b in combinations(countries, 2):
        sub = wide[[a, b]].dropna()
        vals_a, vals_b = sub[a].values, sub[b].values
        lam, idx = [], []
        for i in range(W_Q, len(sub) + 1):
            tau, _ = kendalltau(vals_a[i - W_Q:i], vals_b[i - W_Q:i])
            lam.append(lam_from_tau(tau))
            idx.append(sub.index[i - 1])
        ts[f"{a}-{b}"] = pd.Series(lam, index=idx)
    T = pd.DataFrame(ts)
    pm = T.mean(axis=1).dropna()
    m = mask_full.loc[pm.index].values
    vals = pm.values
    mu_c, mu_s = vals[m == 1].mean(), vals[m == 0].mean()
    obs = mu_c - mu_s
    cnt = sum(
        1 for _ in range(B_PERM)
        if (lambda mm: mm.sum() not in (0, len(vals))
            and vals[mm == 1].mean() - vals[mm == 0].mean() >= obs)(
            np.roll(m, rng.integers(1, len(vals) - 1))))
    p_panel = (cnt + 1) / (B_PERM + 1)
    amp = mu_c / mu_s if mu_s > 0 else np.nan
    pd.DataFrame([{"n_countries": len(countries), "window_q": W_Q,
                   "lambdaU_stable": round(mu_s, 4),
                   "lambdaU_crisis": round(mu_c, 4),
                   "amplification": round(amp, 4),
                   "panel_perm_p": round(p_panel, 4)}]).to_csv(
        os.path.join(TAB, "table_quarterly_h1.csv"), index=False)
    print(f"[H1q] amp = {amp:.3f} (stable {mu_s:.3f} -> crisis {mu_c:.3f}), "
          f"panel perm p = {p_panel:.4f}")

    # ---- H3-style Granger share ----
    bal = wide.dropna()
    rows = []
    for a, b in permutations(countries, 2):
        try:
            res = grangercausalitytests(bal[[b, a]].values, maxlag=2,
                                        verbose=False)
            p = min(res[l][0]["ssr_ftest"][1] for l in [1, 2])
        except Exception:
            p = np.nan
        de_fr = "DE/FR involved" if {a, b} & {"DE", "FR"} else "core8"
        rows.append({"causing": a, "caused": b,
                     "p_min_lag12": round(float(p), 4), "group": de_fr})
    g3 = pd.DataFrame(rows)
    g3.to_csv(os.path.join(TAB, "table_quarterly_h3.csv"), index=False)
    for grp, gg in g3.groupby("group"):
        sig = (gg["p_min_lag12"] < 0.05).sum()
        print(f"[H3q] {grp:14s}: {sig}/{len(gg)} significant "
              f"({sig/len(gg):.0%})")
    defr_in = g3[(g3["group"] == "DE/FR involved")
                 & (g3["p_min_lag12"] < 0.05)]
    print(f"      DE/FR links: " + "; ".join(
        f"{r['causing']}->{r['caused']}" for _, r in defr_in.iterrows()))

    print("\nDONE -- 10_quarterly_robustness.py")


if __name__ == "__main__":
    main()
