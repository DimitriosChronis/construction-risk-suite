"""
Paper 5 — Additional sensitivity analyses
=========================================
(A) Tail-threshold sensitivity of the directional source/receiver
    classification: re-estimates the lead-lag joint-exceedance flows
    at q in {0.85, 0.90, 0.95} to probe the estimator's only tuning
    parameter (TAIL_Q).
(B) CSRI component-weight sensitivity: equal weights vs tilted vs
    PCA-derived weights.

Inputs  (data/processed/): portfolio_type_returns_theoretical.csv,
        dynamic_copula_lambdaU.csv, csri_monthly.csv
Outputs (data/processed/): sens_tailq_roles.csv, sens_csri_weights.csv
        (results/):        table_sens_tailq.csv, table_sens_csri_weights.csv
Run from: paper5-portfolio-contagion/src/
"""
import numpy as np
import pandas as pd
from itertools import permutations
from scipy.stats import rankdata

PROC = "../data/processed/"
RES  = "../results/"
WINDOW = 24                      # must match 03/04
Q_GRID = [0.85, 0.90, 0.95]      # 0.90 = baseline used in the paper
SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# (A) TAIL-THRESHOLD SENSITIVITY OF SOURCE/RECEIVER ROLES
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("(A) Tail-threshold sensitivity of directional roles")
print("=" * 60)

rets = pd.read_csv(PROC + "portfolio_type_returns_theoretical.csv",
                   index_col=0, parse_dates=True).sort_index()
lamU = pd.read_csv(PROC + "dynamic_copula_lambdaU.csv",
                   index_col=0, parse_dates=True).sort_index()
types = list(rets.columns)
idx_of = {t: i for i, t in enumerate(types)}
X = rets.to_numpy()
out_idx = lamU.index


def pseudo_obs(x):
    return rankdata(x, method="average") / (len(x) + 1)


def lead_lag_upper(x_lead, y_foll, q):
    """P(Y_t > q | X_{t-1} > q) on a short window (pseudo-obs inside).
    Identical to src/04_contagion_index.py."""
    if len(x_lead) < 4:
        return np.nan
    u = pseudo_obs(x_lead)
    v = pseudo_obs(y_foll)
    xt_1, yt = u[:-1], v[1:]
    cond = xt_1 > q
    if cond.sum() == 0:
        return 0.0
    return float((yt[cond] > q).mean())


rows = []
for q in Q_GRID:
    dir_cols = [f"{a}->{b}" for (a, b) in permutations(types, 2)]
    lam_dir = pd.DataFrame(index=out_idx, columns=dir_cols, dtype=float)
    for end_date in out_idx:
        end_pos = rets.index.get_loc(end_date)
        win = X[end_pos - WINDOW + 1: end_pos + 1]
        for (a, b) in permutations(types, 2):
            lam_dir.loc[end_date, f"{a}->{b}"] = lead_lag_upper(
                win[:, idx_of[a]], win[:, idx_of[b]], q)
    for t_i in types:
        out_avg = lam_dir[[c for c in dir_cols
                           if c.startswith(f"{t_i}->")]].mean(axis=1).mean()
        in_avg = lam_dir[[c for c in dir_cols
                          if c.endswith(f"->{t_i}")]].mean(axis=1).mean()
        net = out_avg - in_avg
        rows.append({"q": q, "type": t_i, "OUT": round(out_avg, 4),
                     "IN": round(in_avg, 4), "NET": round(net, 4),
                     "role": "source" if net > 0 else "receiver"})
    print(f"  q={q}: done")

sens_q = pd.DataFrame(rows)
sens_q.to_csv(PROC + "sens_tailq_roles.csv", index=False)
piv = sens_q.pivot(index="type", columns="q", values="role")
piv_net = sens_q.pivot(index="type", columns="q", values="NET")
stable_roles = (piv.nunique(axis=1) == 1)
print("\nRole by q:")
print(piv.to_string())
print("\nNET by q:")
print(piv_net.to_string())
print(f"\nRoles stable across q for {stable_roles.sum()}/{len(piv)} types")
tbl = piv_net.copy()
tbl["role_q090"] = piv[0.90]
tbl["stable_across_q"] = stable_roles
tbl.to_csv(RES + "table_sens_tailq.csv")

# ══════════════════════════════════════════════════════════════════════
# (B) CSRI COMPONENT-WEIGHT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("(B) CSRI component-weight sensitivity")
print("=" * 60)

csri = pd.read_csv(PROC + "csri_monthly.csv", index_col=0, parse_dates=True)
zc = [c for c in csri.columns if c.startswith("z_")]
Z = csri[zc].dropna()
print("components:", zc, "| months:", len(Z))

base = Z.mean(axis=1)  # equal weights (the paper's CSRI)

# Exogenous crisis windows (macro calendar, as in the LSTM section)
WINDOWS = {
    "2012 sovereign":       ("2011-07-01", "2012-12-31"),
    "2015 capital controls": ("2015-06-01", "2015-12-31"),
    "2022 Ukraine shock":   ("2022-02-01", "2022-12-31"),
}


def crisis_hits(series):
    """Does the index exceed +1 sigma (of its own distribution) inside
    each exogenous crisis window?"""
    thr = series.mean() + series.std(ddof=0)
    hits = {}
    for name, (a, b) in WINDOWS.items():
        seg = series.loc[a:b]
        hits[name] = bool((seg > thr).any()) if len(seg) else None
    return hits


schemes = {"equal (paper)": np.ones(len(zc)) / len(zc)}
for k, c in enumerate(zc):
    w = np.ones(len(zc)); w[k] = 2.0
    schemes[f"tilt 2x {c[2:]}"] = w / w.sum()

# PCA first-component weights (sign-aligned, normalised to sum 1)
Zs = (Z - Z.mean()) / Z.std(ddof=0)
cov = np.cov(Zs.to_numpy().T)
evals, evecs = np.linalg.eigh(cov)
pc1 = evecs[:, -1]
pc1 = np.abs(pc1) / np.abs(pc1).sum()   # positive, sum-1
schemes["PCA-1"] = pc1
print("PCA-1 weights:", dict(zip([c[2:] for c in zc], np.round(pc1, 3))))

rows = []
for name, w in schemes.items():
    s = (Z * w).sum(axis=1)
    hits = crisis_hits(s)
    rows.append({
        "scheme": name,
        **{f"w_{c[2:]}": round(float(wi), 3) for c, wi in zip(zc, w)},
        "corr_with_equal": round(float(s.corr(base)), 4),
        "mean_crisis_minus_stable": round(float(
            s[csri.loc[s.index, "regime"] == "crisis"].mean()
            - s[csri.loc[s.index, "regime"] == "stable"].mean()), 3),
        **{f"exceeds_1sd_{k.split()[0]}": v for k, v in hits.items()},
    })

sens_w = pd.DataFrame(rows)
sens_w.to_csv(PROC + "sens_csri_weights.csv", index=False)
sens_w.to_csv(RES + "table_sens_csri_weights.csv", index=False)
print("\n" + sens_w.to_string(index=False))
print("\nDONE — outputs: sens_tailq_roles.csv, sens_csri_weights.csv")
