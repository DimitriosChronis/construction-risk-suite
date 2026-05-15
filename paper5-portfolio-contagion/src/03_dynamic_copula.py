"""
Paper 5 — Dynamic Copula
===================================
Core methodological contribution of P5.

Fits a TIME-VARYING bivariate Gumbel copula on every pair of project-type
return series, using a rolling 24-month window.  Extracts the upper tail
dependence coefficient λ_U(i,j,t) per pair per month — this is the primitive
quantity that Scripts 04 (contagion index) and 05 (portfolio ES) consume.

Why Gumbel via Kendall's tau?
  - Gumbel copula has closed-form upper tail dependence: λ_U = 2 - 2^(1/θ)
  - Kendall's tau maps to Gumbel parameter:            θ = 1/(1 - τ)
  - Therefore:                                         λ_U = 2 - 2^(1 - τ)
  This avoids MLE instability on 24-point windows while remaining a genuine
  parametric copula (standard in the finance/insurance literature — see
  Embrechts, Lindskog & McNeil 2003).  We fall back to an empirical joint-
  exceedance estimator as a non-parametric sanity check.

Inputs:
  - data/processed/portfolio_type_returns_theoretical.csv   (from 02)

Outputs:
  - data/processed/dynamic_copula_tau.csv        (rolling Kendall τ per pair)
  - data/processed/dynamic_copula_lambdaU.csv    (Gumbel-implied λ_U per pair)
  - data/processed/dynamic_copula_empirical_lambdaU.csv  (non-param sanity)
  - data/processed/dynamic_copula_network_avg.csv  (date, avg λ_U across pairs)
  - data/processed/dynamic_copula_regimes.csv    (stable/crisis label per month)
  - results/table2_copula_regimes.csv            (paper Table 2)

Run from: construction-risk-suite/paper5-portfolio-contagion/src/
"""

import io
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
INPUT       = os.path.join(PROC_DIR, "portfolio_type_returns_theoretical.csv")
WINDOW      = 24                      # rolling window length (months)
CRISIS_Q    = 0.75                    # network-avg λ_U percentile → crisis flag
TAIL_Q      = 0.90                    # threshold for empirical joint-exceedance
EPS         = 1e-6                    # numerical safety for τ near 1

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — Dynamic Copula (time-varying λ_U)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading type-level return series")
if not os.path.exists(INPUT):
    sys.exit(f"  ✗ Missing {INPUT} — run 02_portfolio_construction.py first")

rets = pd.read_csv(INPUT, index_col=0, parse_dates=True).sort_index()
rets = rets.dropna(how="all")
types = list(rets.columns)
n_pairs = len(types) * (len(types) - 1) // 2
print(f"  Types: {types}")
print(f"  Months: {len(rets)}  ({rets.index.min():%Y-%m} → {rets.index.max():%Y-%m})")
print(f"  Pairs: {n_pairs}")

if len(rets) < WINDOW + 1:
    sys.exit(f"  ✗ Need ≥{WINDOW + 1} months for a {WINDOW}-month rolling window "
             f"(have {len(rets)}). Re-run 02 after 01 full download.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Convert to uniform pseudo-observations via empirical CDF (rank/(n+1))."""
    return rankdata(x, method="average") / (len(x) + 1)


def gumbel_lambda_U_from_tau(tau: float) -> float:
    """Upper tail dependence of Gumbel copula implied by Kendall's τ."""
    if tau is None or np.isnan(tau) or tau <= 0:
        return 0.0
    tau_c = min(tau, 1 - EPS)
    return 2.0 - 2.0 ** (1.0 - tau_c)


def empirical_lambda_U(x: np.ndarray, y: np.ndarray, q: float) -> float:
    """Non-parametric upper tail dependence:  P(U>q | V>q) ≈ P(U>q, V>q) / (1-q)."""
    u, v = pseudo_obs(x), pseudo_obs(y)
    joint = np.mean((u > q) & (v > q))
    return float(joint / (1.0 - q)) if (1.0 - q) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROLLING PAIRWISE COPULA ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 2: Rolling {WINDOW}-month copula estimation")

pairs = list(combinations(types, 2))
pair_names = [f"{a}__{b}" for (a, b) in pairs]

out_idx = rets.index[WINDOW - 1:]      # windows are right-aligned
tau_df     = pd.DataFrame(index=out_idx, columns=pair_names, dtype=float)
lamU_df    = pd.DataFrame(index=out_idx, columns=pair_names, dtype=float)
lamU_emp   = pd.DataFrame(index=out_idx, columns=pair_names, dtype=float)

X = rets.to_numpy()
col_idx = {t: i for i, t in enumerate(types)}

for end in range(WINDOW - 1, len(rets)):
    win = X[end - WINDOW + 1: end + 1]        # (WINDOW × 5)
    date = rets.index[end]
    for (a, b), name in zip(pairs, pair_names):
        xa = win[:, col_idx[a]]
        xb = win[:, col_idx[b]]
        if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
            tau = np.nan
        else:
            tau, _ = kendalltau(xa, xb, nan_policy="omit")
        tau_df.loc[date, name] = tau
        lamU_df.loc[date, name] = gumbel_lambda_U_from_tau(tau)
        lamU_emp.loc[date, name] = empirical_lambda_U(xa, xb, TAIL_Q)

print(f"  Estimated {len(out_idx)} monthly snapshots × {len(pairs)} pairs "
      f"= {len(out_idx) * len(pairs):,} copula fits")
print(f"  Mean τ across pairs (full sample): {tau_df.stack().mean():+.3f}")
print(f"  Mean λ_U (Gumbel):                 {lamU_df.stack().mean():+.3f}")
print(f"  Mean λ_U (empirical, q={TAIL_Q}):   {lamU_emp.stack().mean():+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. NETWORK-AVERAGE λ_U + REGIME CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 3: Network-average λ_U & regime classification")

network = pd.DataFrame({
    "avg_tau":         tau_df.mean(axis=1),
    "avg_lambdaU":     lamU_df.mean(axis=1),
    "avg_lambdaU_emp": lamU_emp.mean(axis=1),
})
network.index.name = "date"

# Endogenous regime: crisis when avg λ_U > CRISIS_Q percentile of its history
thr = network["avg_lambdaU"].quantile(CRISIS_Q)
network["regime"] = np.where(network["avg_lambdaU"] > thr, "crisis", "stable")

# Exogenous regime (Greek macro calendar) — for Table 2 benchmarking
def exo_regime(d: pd.Timestamp) -> str:
    y = d.year
    if 2010 <= y <= 2016: return "GR_crisis"
    if 2017 <= y <= 2019: return "recovery"
    if 2020 <= y <= 2022: return "covid_energy"
    return "post_2022"


network["regime_exo"] = [exo_regime(d) for d in network.index]

reg_counts = network["regime"].value_counts().to_dict()
print(f"  Endogenous regimes (threshold λ_U>{thr:.3f}): {reg_counts}")
print(f"  Exogenous regimes (by calendar):")
print(network["regime_exo"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 5. TABLE 2 (per-pair regime comparison: stable vs crisis)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 4: Building Table 2 (λ_U stable vs crisis, per pair)")

table2_rows = []
for name in pair_names:
    stable_vals = lamU_df.loc[network["regime"] == "stable", name].dropna()
    crisis_vals = lamU_df.loc[network["regime"] == "crisis", name].dropna()
    stable_mean = stable_vals.mean() if len(stable_vals) else np.nan
    crisis_mean = crisis_vals.mean() if len(crisis_vals) else np.nan
    amp = (crisis_mean / stable_mean) if stable_mean and stable_mean > 0 else np.nan
    table2_rows.append({
        "pair":             name,
        "lambdaU_stable":   stable_mean,
        "lambdaU_crisis":   crisis_mean,
        "amplification":    amp,
        "n_stable_months":  int(len(stable_vals)),
        "n_crisis_months":  int(len(crisis_vals)),
    })
table2 = pd.DataFrame(table2_rows).sort_values("amplification", ascending=False)
print(table2.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Saving")

files = {
    "dynamic_copula_tau.csv":               tau_df,
    "dynamic_copula_lambdaU.csv":           lamU_df,
    "dynamic_copula_empirical_lambdaU.csv": lamU_emp,
    "dynamic_copula_network_avg.csv":       network,
}
for f, df in files.items():
    p = os.path.join(PROC_DIR, f)
    df.to_csv(p)
    print(f"  Saved: {p}")

regimes_path = os.path.join(PROC_DIR, "dynamic_copula_regimes.csv")
network[["regime", "regime_exo"]].to_csv(regimes_path)
print(f"  Saved: {regimes_path}")

table2_path = os.path.join(RESULTS_DIR, "table2_copula_regimes.csv")
table2.to_csv(table2_path, index=False)
print(f"  Saved: {table2_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Dynamic copula fitted")
print("=" * 64)
print(f"  Period covered:    {out_idx.min():%Y-%m} → {out_idx.max():%Y-%m}")
print(f"  Months:            {len(out_idx)}")
print(f"  Pairs:             {len(pairs)}")
print(f"  Network avg λ_U:   full = {network['avg_lambdaU'].mean():.3f}  |  "
      f"stable = {network.loc[network['regime']=='stable','avg_lambdaU'].mean():.3f}  |  "
      f"crisis = {network.loc[network['regime']=='crisis','avg_lambdaU'].mean():.3f}")
print(f"\nNext: run 04_contagion_index.py")
print("=" * 64)
