"""
Paper 5 — Contagion Index
===================================
Second methodological contribution of P5.

Builds a time-varying CONTAGION INDEX per project type from the pairwise
λ_U(i,j,t) produced by 03_dynamic_copula.py, and classifies each type as
SOURCE (lead) or RECEIVER (follow) of systemic tail co-movement.

Contagion index (undirected):
    CI_i(t) = (1 / (K-1)) · Σ_{j≠i}  λ_U(i,j,t)
    → average upper tail dependence of type i with every other type.

Directional lead-lag (rolling empirical joint exceedance):
    λ_U^lead(i→j,t) = P( pseudo-U_j(t) > q | pseudo-U_i(t-1) > q )
    estimated over the same rolling window as 03.

Net flow:
    OUT_i(t) = mean_{j≠i} λ_U^lead(i→j,t)    (i leads others)
    IN_i(t)  = mean_{j≠i} λ_U^lead(j→i,t)    (i follows others)
    NET_i(t) = OUT_i(t) - IN_i(t)            (>0 ⇒ source, <0 ⇒ receiver)

Crisis amplification factor:
    AF_i = mean CI_i during 'crisis' regime / mean CI_i during 'stable' regime

Inputs:
  - data/processed/portfolio_type_returns_theoretical.csv    (from 02)
  - data/processed/dynamic_copula_lambdaU.csv                (from 03)
  - data/processed/dynamic_copula_regimes.csv                (from 03)

Outputs:
  - data/processed/contagion_index.csv          (months × type, CI_i(t))
  - data/processed/contagion_flows.csv          (months × [OUT/IN/NET]_type)
  - data/processed/contagion_classification.csv (per type: source/receiver)
  - results/table3_contagion_by_type.csv        (paper Table 3)

Run from: construction-risk-suite/paper5-portfolio-contagion/src/
"""

import io
import os
import sys
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
WINDOW      = 24          # rolling window (must match 03)
TAIL_Q      = 0.90        # tail threshold for directional estimator
EPS         = 1e-9

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — Contagion Index (source ↔ receiver network)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading inputs")

rets_path    = os.path.join(PROC_DIR, "portfolio_type_returns_theoretical.csv")
lamU_path    = os.path.join(PROC_DIR, "dynamic_copula_lambdaU.csv")
regimes_path = os.path.join(PROC_DIR, "dynamic_copula_regimes.csv")

for p in (rets_path, lamU_path, regimes_path):
    if not os.path.exists(p):
        sys.exit(f"  ✗ Missing {p} — run 02 and 03 first")

rets    = pd.read_csv(rets_path, index_col=0, parse_dates=True).sort_index()
lamU    = pd.read_csv(lamU_path, index_col=0, parse_dates=True).sort_index()
regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True).sort_index()

types = list(rets.columns)
K = len(types)
pairs = list(combinations(types, 2))
print(f"  Types: {types}")
print(f"  λ_U pairs: {len(pairs)}  |  months in copula file: {len(lamU)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNDIRECTED CONTAGION INDEX CI_i(t)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Computing undirected contagion index CI_i(t)")

CI = pd.DataFrame(index=lamU.index, columns=types, dtype=float)
for i, t_i in enumerate(types):
    related = [c for c in lamU.columns
               if c.startswith(f"{t_i}__") or c.endswith(f"__{t_i}")]
    CI[t_i] = lamU[related].mean(axis=1)

print(f"  CI range: [{CI.min().min():.3f}, {CI.max().max():.3f}]")
print(f"  Mean CI per type (full sample):")
for t in types:
    print(f"    {t:9s}: {CI[t].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIRECTIONAL LEAD-LAG (rolling empirical joint exceedance)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 3: Directional lead-lag λ_U^lead(i→j,t)  (rolling {WINDOW}M, q={TAIL_Q})")


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / (len(x) + 1)


def lead_lag_upper(x_lead: np.ndarray, y_foll: np.ndarray, q: float) -> float:
    """P(Y_t > q | X_{t-1} > q) on a short window (pseudo-obs inside)."""
    if len(x_lead) < 4:
        return np.nan
    u = pseudo_obs(x_lead)                # t-1 pseudo-obs of leader
    v = pseudo_obs(y_foll)                # t   pseudo-obs of follower
    xt_1 = u[:-1]                         # leader at t-1
    yt   = v[1:]                          # follower at t
    cond = xt_1 > q
    if cond.sum() == 0:
        return 0.0
    return float((yt[cond] > q).mean())


X = rets.to_numpy()
idx_of = {t: i for i, t in enumerate(types)}

# Window ends at same dates as lamU
out_idx = lamU.index
dir_cols = [f"{a}->{b}" for (a, b) in permutations(types, 2)]
lamU_dir = pd.DataFrame(index=out_idx, columns=dir_cols, dtype=float)

for end_date in out_idx:
    end_pos = rets.index.get_loc(end_date)
    start   = end_pos - WINDOW + 1
    win = X[start: end_pos + 1]           # (WINDOW × K)
    for (a, b) in permutations(types, 2):
        xa = win[:, idx_of[a]]            # leader
        xb = win[:, idx_of[b]]            # follower
        lamU_dir.loc[end_date, f"{a}->{b}"] = lead_lag_upper(xa, xb, TAIL_Q)

print(f"  Estimated {len(out_idx) * len(dir_cols):,} directional pair-months")
print(f"  Mean directional intensity: {lamU_dir.stack().mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OUT / IN / NET FLOWS  ⇒  SOURCE vs RECEIVER
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Computing OUT / IN / NET flows")

flows = pd.DataFrame(index=out_idx)
for t_i in types:
    out_cols = [c for c in dir_cols if c.startswith(f"{t_i}->")]
    in_cols  = [c for c in dir_cols if c.endswith(f"->{t_i}")]
    flows[f"OUT_{t_i}"] = lamU_dir[out_cols].mean(axis=1)
    flows[f"IN_{t_i}"]  = lamU_dir[in_cols].mean(axis=1)
    flows[f"NET_{t_i}"] = flows[f"OUT_{t_i}"] - flows[f"IN_{t_i}"]
flows.index.name = "date"


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION + REGIME AMPLIFICATION (Table 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Building Table 3 (contagion by type)")

# Align regime labels
regimes_al = regimes.reindex(out_idx)
is_crisis  = regimes_al["regime"] == "crisis"
is_stable  = regimes_al["regime"] == "stable"

table3_rows = []
for t_i in types:
    ci_full   = CI[t_i].mean()
    ci_stable = CI.loc[is_stable, t_i].mean()
    ci_crisis = CI.loc[is_crisis, t_i].mean()
    amp       = (ci_crisis / ci_stable) if ci_stable and ci_stable > 0 else np.nan

    out_avg = flows[f"OUT_{t_i}"].mean()
    in_avg  = flows[f"IN_{t_i}"].mean()
    net_avg = flows[f"NET_{t_i}"].mean()
    role    = "source" if net_avg > 0 else "receiver"

    table3_rows.append({
        "type":            t_i,
        "CI_full":         ci_full,
        "CI_stable":       ci_stable,
        "CI_crisis":       ci_crisis,
        "amplification":   amp,
        "OUT":             out_avg,
        "IN":              in_avg,
        "NET":             net_avg,
        "role":            role,
    })

table3 = pd.DataFrame(table3_rows).sort_values("NET", ascending=False)
print(table3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Saving")

CI.to_csv(os.path.join(PROC_DIR, "contagion_index.csv"))
flows.to_csv(os.path.join(PROC_DIR, "contagion_flows.csv"))
table3.to_csv(os.path.join(PROC_DIR, "contagion_classification.csv"), index=False)
table3.to_csv(os.path.join(RESULTS_DIR, "table3_contagion_by_type.csv"), index=False)

for f in ("contagion_index.csv", "contagion_flows.csv",
          "contagion_classification.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table3_contagion_by_type.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Contagion index built")
print("=" * 64)
sources   = [r["type"] for r in table3_rows if r["role"] == "source"]
receivers = [r["type"] for r in table3_rows if r["role"] == "receiver"]
print(f"  Sources  (net λ_U outflow > 0): {sources}")
print(f"  Receivers (net λ_U inflow  > 0): {receivers}")
print(f"  Max crisis amplification: "
      f"{max(r['amplification'] for r in table3_rows if not np.isnan(r['amplification'])):.3f} "
      f"({max(table3_rows, key=lambda r: r['amplification'] if not np.isnan(r['amplification']) else -1)['type']})")
print(f"\nNext: run 05_portfolio_es.py")
print("=" * 64)
