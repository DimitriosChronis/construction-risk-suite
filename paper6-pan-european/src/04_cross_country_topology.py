"""
04_cross_country_topology.py
============================
H1: cross-border tail dependence between country CCI series.

For each of the 28 country pairs of the primary panel, a rolling
pair-Gumbel copula is estimated on log-returns via the Kendall-tau
inversion (theta = 1/(1-tau), lambda_U = 2 - 2^(1/theta); tau <= 0
maps to lambda_U = 0), exactly the estimator family of the published
companion work. Crisis-vs-stable amplification uses the four
PRE-REGISTERED exogenous EU windows; inference
uses a circular block permutation of the crisis mask (B = 1000,
block = 12), which preserves serial correlation under the null.

Sensitivity built in:
  * rolling window w in {24, 36} months;
  * Clayton lower-tail contrast lambda_L (asymmetry check:
    upper-tail amplification should exceed lower-tail).

Inputs : data/processed/cci_features_primary.csv
Outputs: data/processed/lambdaU_pairs_w{24,36}.csv   (time series)
         results/tables/table_cross_country_lambdaU.csv (per pair)
         results/tables/table_h1_summary.csv
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

WINDOWS = [24, 36]
B_PERM = 1000
BLOCK = 12
SEED = 42
MEDITERRANEAN = {"GR", "ES", "IT", "PT"}


def lambda_from_tau(tau, upper=True):
    if tau is None or np.isnan(tau) or tau <= 0:
        return 0.0
    tau = min(tau, 0.99)
    if upper:                      # Gumbel upper tail
        theta = 1.0 / (1.0 - tau)
        return 2.0 - 2.0 ** (1.0 / theta)
    theta = 2.0 * tau / (1.0 - tau)   # Clayton lower tail
    return 2.0 ** (-1.0 / theta) if theta > 0 else 0.0


def rolling_lambda(x, y, dates, w):
    lamU, lamL, out_dates = [], [], []
    for i in range(w, len(x) + 1):
        xa, ya = x[i - w:i], y[i - w:i]
        if np.isnan(xa).any() or np.isnan(ya).any():
            continue
        tau, _ = kendalltau(xa, ya)
        lamU.append(lambda_from_tau(tau, upper=True))
        lamL.append(lambda_from_tau(tau, upper=False))
        out_dates.append(dates[i - 1])
    return out_dates, lamU, lamL


def block_perm_pvalue(series, mask, observed_diff, rng):
    """Circular block rotation of the crisis mask; two-sided-ish
    one-tail p for diff = mean(crisis) - mean(stable) > 0."""
    n = len(series)
    count = 0
    for _ in range(B_PERM):
        shift = rng.integers(1, n - 1)
        m = np.roll(mask, shift)
        if m.sum() in (0, n):
            continue
        d = series[m == 1].mean() - series[m == 0].mean()
        if d >= observed_diff:
            count += 1
    return (count + 1) / (B_PERM + 1)


def main():
    os.makedirs(TAB, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- H1 cross-country tail dependence")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    countries = sorted(feat["country"].unique())
    wide_r = feat.pivot(index="date", columns="country",
                        values="log_return").sort_index()
    wide_cx = feat.pivot(index="date", columns="country",
                         values="crisis_exog").sort_index()

    pair_rows, summary_rows = [], []
    for w in WINDOWS:
        ts_frames = {}
        amps = []
        for a, b in combinations(countries, 2):
            sub = wide_r[[a, b]].dropna()
            dates = sub.index.to_list()
            d, lamU, lamL = rolling_lambda(sub[a].values, sub[b].values,
                                           dates, w)
            s = pd.Series(lamU, index=pd.DatetimeIndex(d))
            ts_frames[f"{a}-{b}"] = s
            mask = wide_cx.loc[s.index, [a, b]].max(axis=1).values
            lam = s.values
            mu_c = lam[mask == 1].mean() if (mask == 1).any() else np.nan
            mu_s = lam[mask == 0].mean()
            diff = mu_c - mu_s
            p = block_perm_pvalue(lam, mask, diff, rng)
            sL = np.array(lamL)
            muL_c = sL[mask == 1].mean() if (mask == 1).any() else np.nan
            muL_s = sL[mask == 0].mean()
            bloc = ("MED-MED" if {a, b} <= MEDITERRANEAN else
                    "N-N" if not ({a, b} & MEDITERRANEAN) else "MED-N")
            amp = mu_c / mu_s if mu_s > 0 else np.nan
            amps.append(amp)
            pair_rows.append({
                "window": w, "pair": f"{a}-{b}", "bloc": bloc,
                "lambdaU_stable": round(mu_s, 4),
                "lambdaU_crisis": round(mu_c, 4),
                "amplification": round(amp, 4) if amp == amp else np.nan,
                "perm_p": round(p, 4),
                "lambdaL_stable": round(muL_s, 4),
                "lambdaL_crisis": round(muL_c, 4),
                "n_months": len(s),
            })
        pd.DataFrame(ts_frames).to_csv(
            os.path.join(PROC, f"lambdaU_pairs_w{w}.csv"))

        pr = pd.DataFrame([r for r in pair_rows if r["window"] == w])
        sig = (pr["perm_p"] < 0.05).sum()
        mean_amp = np.nanmean(amps)

        # PANEL-LEVEL test: rotate the exogenous mask by the SAME shift
        # for every pair (preserves cross-pair dependence), and compare
        # the pooled crisis-minus-stable difference of the panel-mean
        # lambda_U series against its null distribution.
        ts = pd.DataFrame(ts_frames)
        panel_mean = ts.mean(axis=1).dropna()
        exog_any = wide_cx.loc[panel_mean.index].max(axis=1).values
        vals = panel_mean.values
        obs = vals[exog_any == 1].mean() - vals[exog_any == 0].mean()
        cnt = 0
        for _ in range(B_PERM):
            m = np.roll(exog_any, rng.integers(1, len(vals) - 1))
            if m.sum() in (0, len(vals)):
                continue
            if vals[m == 1].mean() - vals[m == 0].mean() >= obs:
                cnt += 1
        p_panel = (cnt + 1) / (B_PERM + 1)

        summary_rows.append({
            "window": w,
            "mean_amplification": round(mean_amp, 4),
            "pairs_amp_gt_1": int((pr["amplification"] > 1).sum()),
            "pairs_sig_p05": int(sig),
            "n_pairs": len(pr),
            "mean_lambdaU_stable": round(pr["lambdaU_stable"].mean(), 4),
            "mean_lambdaU_crisis": round(pr["lambdaU_crisis"].mean(), 4),
            "panel_diff": round(obs, 4),
            "panel_perm_p": round(p_panel, 4),
        })
        print(f"[w={w}] mean amplification = {mean_amp:.3f} | "
              f"amp>1: {(pr['amplification'] > 1).sum()}/{len(pr)} pairs | "
              f"per-pair p<0.05: {sig}/{len(pr)} | "
              f"PANEL diff = {obs:+.4f}, perm p = {p_panel:.4f}")
        for blc, g in pr.groupby("bloc"):
            print(f"    {blc:7s} mean amp {np.nanmean(g['amplification']):.3f}"
                  f"  (n={len(g)})")

    pd.DataFrame(pair_rows).to_csv(
        os.path.join(TAB, "table_cross_country_lambdaU.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(TAB, "table_h1_summary.csv"), index=False)
    print("\nDONE -- 04_cross_country_topology.py")


if __name__ == "__main__":
    main()
