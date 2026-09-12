"""
13_extended_robustness.py
=========================
Extended robustness battery: three additional sensitivity analyses
for the headline results.

(R-A) Endogenous-label definition sweep: crisis percentile
      {0.67, 0.70, 0.75, 0.80, 0.85} x vol window {3, 6, 12}.
      For each combo: ALL and MED-MED panel amplification under the
      endogenous labels (point-in-time throughout). Shows the H1
      subgroup finding is not an artifact of P75/6M.
(R-B) Seasonality robustness: the Eurostat series are NSA. Each
      country's log-returns are deseasonalised by removing monthly
      means (calendar-dummy regression), and the H1 exogenous-window
      panel amplification is recomputed for w = 24.
(R-C) GR-source robustness: the H3 quarterly Granger network re-run
      with the INDEPENDENT Eurostat EL quarterly COST series in
      place of the ELSTAT monthly aggregate, to show Greece's
      transmitter role is not an artifact of the national index.

Outputs: results/tables/table_robustness_labels.csv
         results/tables/table_robustness_seasonal.csv
         results/tables/table_robustness_gr_source.csv
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

MED = {"GR", "ES", "IT", "PT"}
MIN_HIST = 36
W = 24
B_PERM = 500
SEED = 42


def lam_from_tau(tau):
    if tau is None or np.isnan(tau) or tau <= 0:
        return 0.0
    theta = 1.0 / (1.0 - min(tau, 0.99))
    return 2.0 - 2.0 ** (1.0 / theta)


def pit_labels(sigma, pct):
    lab = np.full(len(sigma), np.nan)
    v = sigma.values
    for i in range(len(v)):
        h = v[: i + 1]
        h = h[~np.isnan(h)]
        if len(h) < MIN_HIST or np.isnan(v[i]):
            continue
        lab[i] = 1.0 if v[i] > np.quantile(h, pct) else 0.0
    return lab


def panel_amp(ts, mask_series, pairs, rng):
    pm = ts[pairs].mean(axis=1).dropna()
    m = mask_series.reindex(pm.index).fillna(0).astype(int).values
    v = pm.values
    if m.sum() in (0, len(v)):
        return np.nan, np.nan
    mu_c, mu_s = v[m == 1].mean(), v[m == 0].mean()
    obs = mu_c - mu_s
    cnt = 0
    for _ in range(B_PERM):
        mm = np.roll(m, rng.integers(1, len(v) - 1))
        if mm.sum() in (0, len(v)):
            continue
        if v[mm == 1].mean() - v[mm == 0].mean() >= obs:
            cnt += 1
    return (mu_c / mu_s if mu_s > 0 else np.nan,
            (cnt + 1) / (B_PERM + 1))


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- extended robustness battery")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index()
    ts24 = pd.read_csv(os.path.join(PROC, "lambdaU_pairs_w24.csv"),
                       index_col=0, parse_dates=True)
    med_pairs = [c for c in ts24.columns
                 if set(c.split("-")) <= MED]

    # ---------------- (R-A) label-definition sweep -------------------
    print("\n[R-A] endogenous label sweep (pct x vol window):")
    rows = []
    for volw in [3, 6, 12]:
        sig = {c: wide[c].rolling(volw).std() for c in wide.columns}
        for pct in [0.67, 0.70, 0.75, 0.80, 0.85]:
            lab = pd.DataFrame(
                {c: pit_labels(sig[c], pct) for c in wide.columns},
                index=wide.index).fillna(0)
            mask_all = (lab.max(axis=1) > 0).astype(int)
            mask_med = (lab[list(MED)].max(axis=1) > 0).astype(int)
            ampA, pA = panel_amp(ts24, mask_all, list(ts24.columns), rng)
            ampM, pM = panel_amp(ts24, mask_med, med_pairs, rng)
            rows.append({"vol_window": volw, "crisis_pct": pct,
                         "ALL_amp": round(ampA, 3), "ALL_p": round(pA, 3),
                         "MEDMED_amp": round(ampM, 3),
                         "MEDMED_p": round(pM, 3)})
            print(f"  vol={volw:2d} pct={pct:.2f}  "
                  f"ALL amp={ampA:5.3f} (p={pA:.3f})  "
                  f"MED-MED amp={ampM:5.3f} (p={pM:.3f})")
    pd.DataFrame(rows).to_csv(
        os.path.join(TAB, "table_robustness_labels.csv"), index=False)

    # ---------------- (R-B) seasonality robustness -------------------
    print("\n[R-B] deseasonalised H1 (monthly-dummy residuals, w=24):")
    des = wide.copy()
    for c in des.columns:
        mm = des.index.month
        for k in range(1, 13):
            sel = mm == k
            des.loc[sel, c] = des.loc[sel, c] - des.loc[sel, c].mean()
    ts = {}
    for a, b in combinations(sorted(des.columns), 2):
        sub = des[[a, b]].dropna()
        va, vb = sub[a].values, sub[b].values
        lam, idx = [], []
        for i in range(W, len(sub) + 1):
            tau, _ = kendalltau(va[i - W:i], vb[i - W:i])
            lam.append(lam_from_tau(tau))
            idx.append(sub.index[i - 1])
        ts[f"{a}-{b}"] = pd.Series(lam, index=idx)
    Tdes = pd.DataFrame(ts)
    cx = feat.pivot(index="date", columns="country",
                    values="crisis_exog").sort_index()
    mask_exog = (cx.max(axis=1) > 0).astype(int)
    med_p = [c for c in Tdes.columns if set(c.split("-")) <= MED]
    ampA, pA = panel_amp(Tdes, mask_exog, list(Tdes.columns), rng)
    ampM, pM = panel_amp(Tdes, mask_exog, med_p, rng)
    pd.DataFrame([{"variant": "deseasonalised",
                   "ALL_amp": round(ampA, 3), "ALL_p": round(pA, 3),
                   "MEDMED_amp": round(ampM, 3),
                   "MEDMED_p": round(pM, 3)}]).to_csv(
        os.path.join(TAB, "table_robustness_seasonal.csv"), index=False)
    print(f"  ALL amp={ampA:.3f} (p={pA:.3f})  "
          f"MED-MED amp={ampM:.3f} (p={pM:.3f})")

    # ---------------- (R-C) GR source robustness ---------------------
    print("\n[R-C] quarterly Granger with Eurostat EL (not ELSTAT):")
    mon = pd.read_csv(os.path.join(PROC, "cci_panel_primary.csv"),
                      parse_dates=["date"])
    frames = {}
    for c, g in mon.groupby("country"):
        if c == "GR":
            continue
        lvl = g.set_index("date")["level"].resample("QE").mean()
        frames[c] = np.log(lvl).diff()
    q = pd.read_csv(os.path.join(RAW, "eurostat_cci_quarterly.csv"),
                    parse_dates=["date"])
    el = q[(q["country"] == "EL") & (q["indicator"] == "COST")]
    lvl = el.set_index("date")["level"].resample("QE").last()
    frames["GR"] = np.log(lvl).diff()
    wq = pd.DataFrame(frames).dropna()
    rows_c = []
    for a, b in permutations(sorted(wq.columns), 2):
        try:
            res = grangercausalitytests(wq[[b, a]].values, maxlag=2,
                                        verbose=False)
            p = min(res[l][0]["ssr_ftest"][1] for l in [1, 2])
        except Exception:
            p = np.nan
        rows_c.append({"causing": a, "caused": b,
                       "p_min_lag12": round(float(p), 4)})
    gq = pd.DataFrame(rows_c)
    gq.to_csv(os.path.join(TAB, "table_robustness_gr_source.csv"),
              index=False)
    out_deg = gq[(gq["causing"] == "GR")
                 & (gq["p_min_lag12"] < 0.05)].shape[0]
    in_deg = gq[(gq["caused"] == "GR")
                & (gq["p_min_lag12"] < 0.05)].shape[0]
    tot_sig = (gq["p_min_lag12"] < 0.05).sum()
    print(f"  Eurostat-EL: GR out-degree {out_deg}/7, in-degree "
          f"{in_deg}/7 | network sig share "
          f"{tot_sig}/{len(gq)} ({tot_sig/len(gq):.0%})")

    print("\nDONE -- 13_extended_robustness.py")


if __name__ == "__main__":
    main()
