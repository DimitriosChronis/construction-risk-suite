"""
04c_h1_episodes.py
==================
H1 inference upgrade (post-Phase-G):

(1) PER-EPISODE amplification: for each of the four pre-registered
    EU crisis windows separately, panel-mean lambda_U inside the
    window vs the stable months. Consistency count (amp > 1) with an
    exact one-sided sign test (H0: P(amp>1) = 1/2 per episode).
(2) Moving-block bootstrap CI (B = 2000, block = 12) on the overall
    crisis/stable amplification ratio: (lambda_U, mask) months are
    resampled JOINTLY in blocks, preserving both serial correlation
    and the label alignment, giving a percentile CI to report next
    to the permutation p (CIs reported beside point estimates).

Inputs : data/processed/lambdaU_pairs_w{24,36}.csv,
         data/processed/cci_features_primary.csv
Outputs: results/tables/table_h1_episodes.csv
         results/tables/table_h1_bootstrap_ci.csv
"""

import os
from math import comb

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

EPISODES = {
    "GFC":       ("2008-09-01", "2009-12-31"),
    "SOVEREIGN": ("2011-07-01", "2012-12-31"),
    "COVID":     ("2020-03-01", "2020-12-31"),
    "UKRAINE":   ("2022-02-01", "2022-12-31"),
}
B_BOOT = 2000
BLOCK = 12
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- H1 per-episode analysis + block-bootstrap CI")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    cx = feat.pivot(index="date", columns="country",
                    values="crisis_exog").sort_index()

    ep_rows, ci_rows = [], []
    for w in [24, 36]:
        ts = pd.read_csv(os.path.join(PROC, f"lambdaU_pairs_w{w}.csv"),
                         index_col=0, parse_dates=True)
        pm = ts.mean(axis=1).dropna()
        mask_any = (cx.loc[pm.index].max(axis=1) > 0).astype(int)
        stable_mu = pm[mask_any == 0].mean()

        n_up = 0
        for name, (a, b) in EPISODES.items():
            seg = pm.loc[a:b]
            if seg.empty:
                continue
            amp = seg.mean() / stable_mu
            pctile = (pm < seg.max()).mean()
            n_up += amp > 1
            ep_rows.append({"window": w, "episode": name,
                            "n_months": len(seg),
                            "lambdaU_episode": round(seg.mean(), 4),
                            "lambdaU_stable": round(stable_mu, 4),
                            "amplification": round(amp, 4),
                            "peak_percentile": round(pctile, 3)})
        k = len(EPISODES)
        sign_p = sum(comb(k, i) for i in range(n_up, k + 1)) / 2 ** k
        print(f"[w={w}] episodes with amp>1: {n_up}/{k} "
              f"(exact sign test p = {sign_p:.4f})")
        for r in [r for r in ep_rows if r["window"] == w]:
            print(f"    {r['episode']:9s} amp={r['amplification']:5.3f} "
                  f"peak pct={r['peak_percentile']:.2f}")

        # ---- joint moving-block bootstrap CI on the overall ratio ----
        # block-length sensitivity {6, 12, 18}: the inference choice
        # itself is swept
        vals = pm.values
        mask = mask_any.values
        n = len(vals)
        point = vals[mask == 1].mean() / vals[mask == 0].mean()
        for blk in [6, 12, 18]:
            n_blocks = int(np.ceil(n / blk))
            ratios = []
            for _ in range(B_BOOT):
                starts = rng.integers(0, n - blk + 1, size=n_blocks)
                iv = np.concatenate([np.arange(s, s + blk)
                                     for s in starts])[:n]
                v, m = vals[iv], mask[iv]
                if m.sum() in (0, n):
                    continue
                mu_s = v[m == 0].mean()
                if mu_s <= 0:
                    continue
                ratios.append(v[m == 1].mean() / mu_s)
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            ci_rows.append({"window": w,
                            "amplification": round(point, 4),
                            "ci_lo_95": round(lo, 4),
                            "ci_hi_95": round(hi, 4),
                            "excludes_1": int(lo > 1.0), "B": B_BOOT,
                            "block": blk,
                            "episodes_amp_gt_1": f"{n_up}/{k}",
                            "sign_test_p": round(sign_p, 4)})
            print(f"    overall amp = {point:.3f}, 95% CI (block={blk:2d}) "
                  f"[{lo:.3f}, {hi:.3f}]"
                  f"{'  (excludes 1)' if lo > 1 else ''}")

    pd.DataFrame(ep_rows).to_csv(
        os.path.join(TAB, "table_h1_episodes.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(
        os.path.join(TAB, "table_h1_bootstrap_ci.csv"), index=False)
    print("\nDONE -- 04c_h1_episodes.py")


if __name__ == "__main__":
    main()
