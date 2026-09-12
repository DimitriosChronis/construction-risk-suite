"""
12_copula_families.py
=====================
G3: copula-family sensitivity. For each of the 28 primary pairs and
each regime subsample (stable / exogenous crisis), fits four
one-parameter copulas by MLE on pseudo-observations -- Gumbel,
Clayton, Frank, Gaussian -- and records AIC ranks. Reports (i) how
often the upper-tail family (Gumbel) wins or ties within 2 AIC in
crisis vs stable months, and (ii) the Gumbel-implied lambda_U per
regime (pooled), cross-checking the rolling estimator of H1.

Inputs : data/processed/cci_features_primary.csv
Outputs: results/tables/table_copula_families.csv
         results/tables/table_g3_summary.csv
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pobs(x):
    return stats.rankdata(x) / (len(x) + 1)


def ll_gumbel(theta, u, v):
    if theta <= 1.0001:
        return -1e10
    lu, lv = -np.log(u), -np.log(v)
    A = (lu ** theta + lv ** theta) ** (1 / theta)
    logc = (-A + (theta - 1) * (np.log(lu) + np.log(lv))
            + (1 / theta - 2) * theta * np.log(lu ** theta + lv ** theta)
            / theta + np.log(A + theta - 1) - np.log(u) - np.log(v))
    # use standard density formulation
    t = lu ** theta + lv ** theta
    logc = (-t ** (1 / theta)
            + (theta - 1) * (np.log(lu) + np.log(lv))
            + (1 / theta - 2) * np.log(t)
            + np.log(t ** (1 / theta) + theta - 1)
            + lu + lv)
    return float(np.sum(logc))


def ll_clayton(theta, u, v):
    if theta <= 0.0001:
        return -1e10
    logc = (np.log(1 + theta)
            - (1 + theta) * (np.log(u) + np.log(v))
            - (2 + 1 / theta) * np.log(u ** -theta + v ** -theta - 1))
    return float(np.sum(logc))


def ll_frank(theta, u, v):
    if abs(theta) < 1e-4:
        return -1e10
    et = np.exp(-theta)
    eu, ev = np.exp(-theta * u), np.exp(-theta * v)
    num = theta * (1 - et) * eu * ev
    den = ((1 - et) - (1 - eu) * (1 - ev)) ** 2
    return float(np.sum(np.log(num / den)))


def ll_gauss(rho, u, v):
    if abs(rho) >= 0.999:
        return -1e10
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    logc = (-0.5 * np.log(1 - rho ** 2)
            - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y)
            / (2 * (1 - rho ** 2)))
    return float(np.sum(logc))


FAMILIES = {
    "Gumbel": (ll_gumbel, (1.01, 20.0)),
    "Clayton": (ll_clayton, (0.01, 20.0)),
    "Frank": (ll_frank, (0.05, 40.0)),
    "Gaussian": (ll_gauss, (-0.98, 0.98)),
}


def fit(fam, u, v):
    llf, (lo, hi) = FAMILIES[fam]
    res = optimize.minimize_scalar(lambda t: -llf(t, u, v),
                                   bounds=(lo, hi), method="bounded")
    return float(res.x), -float(res.fun)


def main():
    print("=" * 70)
    print("Paper 6 -- G3 copula-family sensitivity")
    print("=" * 70)
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index()
    cx = feat.pivot(index="date", columns="country",
                    values="crisis_exog").sort_index()

    rows = []
    for a, b in combinations(sorted(wide.columns), 2):
        sub = wide[[a, b]].dropna()
        mask = (cx.loc[sub.index, [a, b]].max(axis=1) > 0).values
        for regime, sel in [("stable", ~mask), ("crisis", mask)]:
            x, y = sub[a].values[sel], sub[b].values[sel]
            if len(x) < 30:
                continue
            u, v = pobs(x), pobs(y)
            fits = {}
            for fam in FAMILIES:
                th, ll = fit(fam, u, v)
                fits[fam] = (th, ll, 2 - 2 * ll)   # k=1 -> AIC=2-2ll
            best = min(fits, key=lambda f: fits[f][2])
            th_g = fits["Gumbel"][0]
            lamU = 2 - 2 ** (1 / th_g)
            rows.append({
                "pair": f"{a}-{b}", "regime": regime, "n": int(sel.sum()),
                "best_family": best,
                "gumbel_within_2AIC": int(
                    fits["Gumbel"][2] - fits[best][2] <= 2),
                "theta_gumbel": round(th_g, 3),
                "lambdaU_gumbel": round(lamU, 4),
                **{f"aic_{f}": round(fits[f][2], 1) for f in FAMILIES},
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "table_copula_families.csv"), index=False)

    summ = []
    for regime, g in df.groupby("regime"):
        summ.append({
            "regime": regime,
            "gumbel_best": int((g["best_family"] == "Gumbel").sum()),
            "gumbel_within2": int(g["gumbel_within_2AIC"].sum()),
            "n_pairs": len(g),
            "mean_lambdaU_pooled": round(g["lambdaU_gumbel"].mean(), 4),
            "best_family_counts": str(
                g["best_family"].value_counts().to_dict()),
        })
        print(f"[{regime:6s}] Gumbel best: "
              f"{(g['best_family'] == 'Gumbel').sum()}/{len(g)} | "
              f"within 2 AIC: {g['gumbel_within_2AIC'].sum()}/{len(g)} | "
              f"pooled mean lambda_U = {g['lambdaU_gumbel'].mean():.3f}")
    s = pd.DataFrame(summ)
    amp = (s.loc[s['regime'] == 'crisis', 'mean_lambdaU_pooled'].iloc[0]
           / s.loc[s['regime'] == 'stable', 'mean_lambdaU_pooled'].iloc[0])
    print(f"pooled crisis/stable lambda_U ratio: {amp:.3f}")
    s.to_csv(os.path.join(TAB, "table_g3_summary.csv"), index=False)

    # Estimator cross-validation: tau-inversion theta vs MLE theta
    # (validates the rolling tau-inversion estimator)
    from scipy.stats import kendalltau
    rows_cv = []
    for (pair, regime), g in df.groupby(["pair", "regime"]):
        a, b = pair.split("-")
        sub = wide[[a, b]].dropna()
        mask = (cx.loc[sub.index, [a, b]].max(axis=1) > 0).values
        sel = mask if regime == "crisis" else ~mask
        tau, _ = kendalltau(sub[a].values[sel], sub[b].values[sel])
        th_tau = 1.0 / (1.0 - tau) if tau and tau > 0 else 1.0
        rows_cv.append({"pair": pair, "regime": regime,
                        "theta_mle": g["theta_gumbel"].iloc[0],
                        "theta_tau": round(th_tau, 3)})
    cv = pd.DataFrame(rows_cv)
    r = np.corrcoef(cv["theta_mle"], cv["theta_tau"])[0, 1]
    cv.to_csv(os.path.join(TAB, "table_estimator_crossval.csv"),
              index=False)
    print(f"tau-inversion vs MLE Gumbel theta: r = {r:.3f} "
          f"(n = {len(cv)} pair-regimes)")
    print("\nDONE -- 12_copula_families.py")


if __name__ == "__main__":
    main()
