"""
04b_h1_subgroups.py
===================
H1 pre-registered fallback tests (B3).

(1) Bloc-level panel permutation: the H1 panel test repeated on the
    MED-MED (6 pairs), N-N (6) and MED-N (16) sub-panels, same
    joint circular rotation of the exogenous crisis mask.
(2) Endogenous-label robustness: amplification re-computed with the
    leak-free point-in-time labels (pair in crisis when EITHER
    country's expanding-percentile label = 1).

Inputs : data/processed/lambdaU_pairs_w{24,36}.csv,
         data/processed/cci_features_primary.csv
Outputs: results/tables/table_h1_subgroups.csv
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

MED = {"GR", "ES", "IT", "PT"}
B_PERM = 1000
SEED = 42


def bloc_of(pair):
    a, b = pair.split("-")
    return ("MED-MED" if {a, b} <= MED else
            "N-N" if not ({a, b} & MED) else "MED-N")


def panel_test(ts, mask_df, pairs, rng, mask_mode):
    sub = ts[pairs].mean(axis=1).dropna()
    if mask_mode == "exog":
        mask = mask_df.loc[sub.index].max(axis=1).values
    else:
        cols = sorted({c for p in pairs for c in p.split("-")})
        mask = (mask_df.loc[sub.index, cols].max(axis=1) > 0).astype(int).values
    vals = sub.values
    if mask.sum() in (0, len(vals)):
        return np.nan, np.nan, np.nan
    mu_c, mu_s = vals[mask == 1].mean(), vals[mask == 0].mean()
    obs = mu_c - mu_s
    cnt = 0
    for _ in range(B_PERM):
        m = np.roll(mask, rng.integers(1, len(vals) - 1))
        if m.sum() in (0, len(vals)):
            continue
        if vals[m == 1].mean() - vals[m == 0].mean() >= obs:
            cnt += 1
    return mu_c / mu_s if mu_s > 0 else np.nan, obs, (cnt + 1) / (B_PERM + 1)


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- H1 subgroup + endogenous-label fallback (B3)")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    cx = feat.pivot(index="date", columns="country",
                    values="crisis_exog").sort_index()
    ce = feat.pivot(index="date", columns="country",
                    values="crisis_endog").sort_index().fillna(0)

    rows = []
    for w in [24, 36]:
        ts = pd.read_csv(os.path.join(PROC, f"lambdaU_pairs_w{w}.csv"),
                         index_col=0, parse_dates=True)
        groups = {"ALL": list(ts.columns)}
        for p in ts.columns:
            groups.setdefault(bloc_of(p), []).append(p)
        for label, mask_df, mode in [("exog", cx, "exog"),
                                     ("endog", ce, "endog")]:
            for gname, pairs in groups.items():
                amp, diff, p = panel_test(ts, mask_df, pairs, rng, mode)
                rows.append({"window": w, "labels": label, "bloc": gname,
                             "n_pairs": len(pairs),
                             "amplification": round(amp, 4),
                             "panel_diff": round(diff, 4),
                             "perm_p": round(p, 4)})
                print(f"[w={w} {label:5s}] {gname:7s} "
                      f"amp={amp:5.3f}  diff={diff:+.4f}  p={p:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(TAB, "table_h1_subgroups.csv"),
                              index=False)
    print("\nDONE -- 04b_h1_subgroups.py")


if __name__ == "__main__":
    main()
