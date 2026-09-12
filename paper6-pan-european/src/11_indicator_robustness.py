"""
11_indicator_robustness.py
==========================
G2: indicator and country-set robustness for the headline tests.

(a) COST-indicator subset (ES, PT, NL, LT, NO -- construction COSTS
    rather than producer prices): H1 panel amplification (w=24).
(b) Extended set (+NO, +PL, non-euro): H1 panel amplification and
    Granger link share.

Inputs : data/processed/cci_features_{cost,extended}.csv
Outputs: results/tables/table_indicator_robustness.csv
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
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

W = 24
B_PERM = 1000
SEED = 42


def lam_from_tau(tau):
    if tau is None or np.isnan(tau) or tau <= 0:
        return 0.0
    theta = 1.0 / (1.0 - min(tau, 0.99))
    return 2.0 - 2.0 ** (1.0 / theta)


def h1_panel(name, rng):
    feat = pd.read_csv(os.path.join(PROC, f"cci_features_{name}.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index()
    cx = feat.pivot(index="date", columns="country",
                    values="crisis_exog").sort_index()
    ts = {}
    for a, b in combinations(sorted(wide.columns), 2):
        sub = wide[[a, b]].dropna()
        lam, idx = [], []
        va, vb = sub[a].values, sub[b].values
        for i in range(W, len(sub) + 1):
            tau, _ = kendalltau(va[i - W:i], vb[i - W:i])
            lam.append(lam_from_tau(tau))
            idx.append(sub.index[i - 1])
        ts[f"{a}-{b}"] = pd.Series(lam, index=idx)
    pm = pd.DataFrame(ts).mean(axis=1).dropna()
    m = (cx.loc[pm.index].max(axis=1) > 0).astype(int).values
    vals = pm.values
    mu_c, mu_s = vals[m == 1].mean(), vals[m == 0].mean()
    obs = mu_c - mu_s
    cnt = 0
    for _ in range(B_PERM):
        mm = np.roll(m, rng.integers(1, len(vals) - 1))
        if mm.sum() in (0, len(vals)):
            continue
        if vals[mm == 1].mean() - vals[mm == 0].mean() >= obs:
            cnt += 1
    p = (cnt + 1) / (B_PERM + 1)

    bal = wide.dropna()
    sig = tot = 0
    for a, b in permutations(bal.columns, 2):
        try:
            res = grangercausalitytests(bal[[b, a]].values, maxlag=6,
                                        verbose=False)
            pv = min(res[l][0]["ssr_ftest"][1] for l in range(1, 7))
            sig += pv < 0.05
            tot += 1
        except Exception:
            pass
    return {"panel": name, "n_countries": wide.shape[1],
            "lambdaU_stable": round(mu_s, 4),
            "lambdaU_crisis": round(mu_c, 4),
            "amplification": round(mu_c / mu_s, 4),
            "panel_perm_p": round(p, 4),
            "granger_sig_share": round(sig / tot, 3) if tot else np.nan}


def h4_extended():
    """H4 rerun on the extended (+NO, +PL) panel."""
    import statsmodels.formula.api as smf
    r = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "data", "raw",
                                 "us_ppi_monthly.csv"),
                    index_col=0, parse_dates=True)
    vol = r.rolling(6).std()
    comp = ((vol - vol.mean()) / vol.std()).mean(axis=1)
    feat = pd.read_csv(os.path.join(PROC, "cci_features_extended.csv"),
                       parse_dates=["date"])
    df = feat[["date", "country", "log_return"]].dropna().copy()
    df["dlncci_pct"] = 100 * df["log_return"]
    df["vol_l1"] = df["date"].map(comp.shift(1))
    df["vol_x_med"] = df["vol_l1"] * df["country"].isin(
        {"GR", "ES", "IT", "PT"}).astype(int)
    df = df.dropna(subset=["vol_l1"])
    m = smf.ols("dlncci_pct ~ vol_l1 + vol_x_med + C(country)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})
    print(f"[H4 extended {df['country'].nunique()} countries] "
          f"b1={m.params['vol_l1']:+.4f} (p={m.pvalues['vol_l1']:.2e})  "
          f"b2={m.params['vol_x_med']:+.4f} "
          f"(p={m.pvalues['vol_x_med']:.3f})")
    return {"panel": "extended_H4", "n_countries": df["country"].nunique(),
            "lambdaU_stable": np.nan, "lambdaU_crisis": np.nan,
            "amplification": np.nan, "panel_perm_p": np.nan,
            "granger_sig_share": np.nan,
            "h4_beta1": round(m.params["vol_l1"], 4),
            "h4_beta2_med": round(m.params["vol_x_med"], 4),
            "h4_beta2_p_cluster": float(f"{m.pvalues['vol_x_med']:.3e}")}


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- G2 indicator / country-set robustness")
    print("=" * 70)
    rows = [h1_panel("cost", rng), h1_panel("extended", rng),
            h4_extended()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "table_indicator_robustness.csv"),
              index=False)
    print(df.to_string(index=False))
    print("\nDONE -- 11_indicator_robustness.py")


if __name__ == "__main__":
    main()
