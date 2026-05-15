"""
Paper 5 — Robustness Suite
===================================
Sensitivity analysis bundle for Q1 reviewer-proofing. Produces four
robustness checks in one pass:

    1. COPULA FAMILY       Gumbel / Student-t / Clayton  →  λ_U comparison
    2. CRISIS THRESHOLD    65 / 70 / 75 / 80 / 85 %      →  DB stability
    3. PORTFOLIO WEIGHTS   equal / budget / inv-vol      →  DB stability
    4. PERMUTATION TEST    shuffle regime labels (10 000 reps)
                           →  p-value for λ_U amplification

All results are written as individual CSVs under data/processed/ and
aggregated into results/table_robustness.csv for the appendix.

Inputs  (all produced by scripts 02, 03, 05):
    portfolio_type_returns_theoretical.csv
    dynamic_copula_tau.csv
    dynamic_copula_network_avg.csv
    dynamic_copula_regimes.csv
    portfolio_metadata.csv
    portfolio_es_by_regime.csv   (for reference only)

Outputs:
    data/processed/robust_copula_family.csv
    data/processed/robust_threshold.csv
    data/processed/robust_weights.csv
    data/processed/robust_permutation.csv
    results/table_robustness.csv          (compact summary for paper)
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, t as student_t

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
ALPHA       = 0.95
WINDOW      = 24
THRESHOLDS  = [0.65, 0.70, 0.75, 0.80, 0.85]
N_PERM      = 10_000
RNG         = np.random.default_rng(42)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — Robustness Suite")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading baseline pipeline outputs")

rets     = pd.read_csv(PROC_DIR + "portfolio_type_returns_theoretical.csv",
                        index_col=0, parse_dates=True).sort_index()
tau_df   = pd.read_csv(PROC_DIR + "dynamic_copula_tau.csv",
                        index_col=0, parse_dates=True).sort_index()
net      = pd.read_csv(PROC_DIR + "dynamic_copula_network_avg.csv",
                        index_col=0, parse_dates=True).sort_index()
regimes  = pd.read_csv(PROC_DIR + "dynamic_copula_regimes.csv",
                        index_col=0, parse_dates=True).sort_index()
meta     = pd.read_csv(PROC_DIR + "portfolio_metadata.csv",
                        encoding="utf-8-sig")

types = list(rets.columns)
K = len(types)
pairs = [(types[i], types[j]) for i in range(K) for j in range(i + 1, K)]
print(f"  Types ({K}): {types}")
print(f"  Pairs ({len(pairs)})")
print(f"  Returns: {len(rets)} months")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — copula λ_U formulas & ES
# ══════════════════════════════════════════════════════════════════════════════
def lambda_U_gumbel(tau: float) -> float:
    """Upper-tail dep. for Gumbel with Kendall's τ (τ ≥ 0)."""
    if np.isnan(tau) or tau <= 0:
        return 0.0
    return 2.0 - 2.0 ** (1.0 - tau)


def lambda_U_clayton(tau: float) -> float:
    """Clayton has NO upper tail dep. → 0 by definition."""
    return 0.0


def lambda_L_clayton(tau: float) -> float:
    """Clayton LOWER tail dep. used as its tail signature."""
    if np.isnan(tau) or tau <= 0:
        return 0.0
    theta = 2.0 * tau / (1.0 - tau)
    return 2.0 ** (-1.0 / theta)


def lambda_U_student(tau: float, nu: float = 4.0) -> float:
    """Student-t copula upper (and lower) tail dependence.
       ρ inferred from τ via ρ = sin(π τ / 2)."""
    if np.isnan(tau):
        return np.nan
    rho = np.sin(np.pi * tau / 2.0)
    rho = float(np.clip(rho, -0.999, 0.999))
    arg = np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * (1.0 - student_t.cdf(arg, df=nu + 1))


def expected_shortfall(x: np.ndarray, alpha: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan
    var_q = np.quantile(x, 1 - alpha)
    tail = x[x <= var_q]
    return float(-tail.mean()) if len(tail) else np.nan


def es_suite(window: pd.DataFrame, weights: np.ndarray, alpha: float) -> dict:
    es_i = np.array([expected_shortfall(window[t].to_numpy(), alpha)
                     for t in window.columns])
    es_sum = float(np.nansum(weights * es_i))
    r_p = window.to_numpy() @ weights
    es_p = expected_shortfall(r_p, alpha)
    db = 1.0 - (es_p / es_sum) if es_sum and es_sum > 0 else np.nan
    return {"es_sum": es_sum, "es_p": es_p,
            "ratio": es_p / es_sum if es_sum and es_sum > 0 else np.nan,
            "db": db}


# ══════════════════════════════════════════════════════════════════════════════
# 1. COPULA FAMILY SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Copula family sensitivity (Gumbel / Student-t / Clayton)")

lamU_g = pd.DataFrame(index=tau_df.index, columns=tau_df.columns, dtype=float)
lamU_t = pd.DataFrame(index=tau_df.index, columns=tau_df.columns, dtype=float)
lamL_c = pd.DataFrame(index=tau_df.index, columns=tau_df.columns, dtype=float)

for col in tau_df.columns:
    lamU_g[col] = tau_df[col].map(lambda tau: lambda_U_gumbel(tau))
    lamU_t[col] = tau_df[col].map(lambda tau: lambda_U_student(tau, nu=4.0))
    lamL_c[col] = tau_df[col].map(lambda tau: lambda_L_clayton(tau))

fam_rows = []
reg_al = regimes.reindex(tau_df.index)["regime"]
for fam_name, mat in (("Gumbel_U", lamU_g),
                       ("StudentT_U(nu=4)", lamU_t),
                       ("Clayton_L", lamL_c)):
    avg = mat.mean(axis=1)
    stable = avg[reg_al == "stable"].mean()
    crisis = avg[reg_al == "crisis"].mean()
    amp = crisis / stable if stable and stable > 0 else np.nan
    fam_rows.append({
        "family":   fam_name,
        "full":     avg.mean(),
        "stable":   stable,
        "crisis":   crisis,
        "amp_c_s":  amp,
    })
robust_family = pd.DataFrame(fam_rows)
print(robust_family.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. CRISIS THRESHOLD SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Crisis threshold sensitivity (65/70/75/80/85 %)")

# Baseline weights (budget) — compute once
type_budget = meta.groupby("type")["budget_eur"].sum().reindex(types).fillna(0)
if type_budget.sum() == 0:
    w_budget = np.full(K, 1.0 / K)
else:
    w_budget = (type_budget / type_budget.sum()).to_numpy()

rets_al = rets.reindex(net.index).dropna()
avg_lam = net["avg_lambdaU"].reindex(rets_al.index)

thr_rows = []
for q in THRESHOLDS:
    cutoff = avg_lam.quantile(q)
    regime_q = np.where(avg_lam >= cutoff, "crisis", "stable")
    regime_q = pd.Series(regime_q, index=rets_al.index)

    stable_m = rets_al.loc[regime_q == "stable"]
    crisis_m = rets_al.loc[regime_q == "crisis"]
    if len(stable_m) < 5 or len(crisis_m) < 5:
        continue
    es_s = es_suite(stable_m, w_budget, ALPHA)
    es_c = es_suite(crisis_m, w_budget, ALPHA)

    # λ_U amp at this threshold
    lam_s = avg_lam[regime_q == "stable"].mean()
    lam_c = avg_lam[regime_q == "crisis"].mean()
    thr_rows.append({
        "threshold_q": q,
        "n_crisis":    int((regime_q == "crisis").sum()),
        "DB_stable":   es_s["db"],
        "DB_crisis":   es_c["db"],
        "DB_drop":     es_s["db"] - es_c["db"],
        "lambdaU_amp": lam_c / lam_s if lam_s > 0 else np.nan,
    })
robust_threshold = pd.DataFrame(thr_rows)
print(robust_threshold.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 3. WEIGHTING SENSITIVITY  (equal / budget / inverse-volatility)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Portfolio weights sensitivity (equal / budget / inv-vol)")

w_equal = np.full(K, 1.0 / K)
# inverse volatility weights
vols = rets[types].std().to_numpy()
w_invvol = (1.0 / vols) / np.sum(1.0 / vols)

reg_al_full = regimes.reindex(rets_al.index)["regime"]
stable_mask = reg_al_full == "stable"
crisis_mask = reg_al_full == "crisis"

wt_rows = []
for wname, w in (("equal", w_equal),
                  ("budget", w_budget),
                  ("inv_vol", w_invvol)):
    for rname, mask in (("full",   pd.Series(True, index=rets_al.index)),
                         ("stable", stable_mask),
                         ("crisis", crisis_mask)):
        sub = rets_al.loc[mask]
        if len(sub) < 5:
            continue
        res = es_suite(sub, w, ALPHA)
        wt_rows.append({
            "weights": wname,
            "regime":  rname,
            "n":       int(len(sub)),
            "ES_sum":  res["es_sum"],
            "ES_P":    res["es_p"],
            "ratio":   res["ratio"],
            "DB":      res["db"],
        })
robust_weights = pd.DataFrame(wt_rows)
print(robust_weights.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n  Weight vectors:")
for wname, w in (("equal", w_equal), ("budget", w_budget), ("inv_vol", w_invvol)):
    parts = [f"{t}={wi:.3f}" for t, wi in zip(types, w)]
    print(f"    {wname:8s}: {', '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PERMUTATION TEST — λ_U amplification
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/4] Permutation test for λ_U amplification  (B={N_PERM})")

lam_series = net["avg_lambdaU"].reindex(rets_al.index).dropna()
reg_series = reg_al_full.reindex(lam_series.index)
common = lam_series.index.intersection(reg_series.dropna().index)
lam_v = lam_series.loc[common].to_numpy()
reg_v = reg_series.loc[common].to_numpy()

n_crisis = int((reg_v == "crisis").sum())
n_stable = int((reg_v == "stable").sum())
if n_crisis < 3 or n_stable < 3:
    print("  ⚠ Not enough crisis/stable months — skipping permutation test")
    robust_perm = pd.DataFrame([{"n_perm": 0, "amp_obs": np.nan,
                                   "p_value": np.nan}])
else:
    amp_obs = lam_v[reg_v == "crisis"].mean() / lam_v[reg_v == "stable"].mean()
    amp_null = np.empty(N_PERM)
    n = len(lam_v)
    for b in range(N_PERM):
        idx = RNG.permutation(n)
        reg_shuf = reg_v[idx]
        m_s = lam_v[reg_shuf == "stable"].mean()
        m_c = lam_v[reg_shuf == "crisis"].mean()
        amp_null[b] = m_c / m_s if m_s > 0 else np.nan
    amp_null = amp_null[~np.isnan(amp_null)]
    # One-sided p-value: P(amp_null >= amp_obs)
    p_value = float((amp_null >= amp_obs).mean())
    print(f"  Observed amp (crisis/stable): {amp_obs:.4f}")
    print(f"  Null mean: {amp_null.mean():.4f}  Null 95%: "
          f"[{np.quantile(amp_null, 0.025):.4f}, {np.quantile(amp_null, 0.975):.4f}]")
    print(f"  One-sided p-value (amp_null ≥ obs): {p_value:.4f}")
    robust_perm = pd.DataFrame([{
        "n_perm":          int(len(amp_null)),
        "n_crisis":        n_crisis,
        "n_stable":        n_stable,
        "amp_observed":    amp_obs,
        "amp_null_mean":   float(amp_null.mean()),
        "amp_null_ci_lo":  float(np.quantile(amp_null, 0.025)),
        "amp_null_ci_hi":  float(np.quantile(amp_null, 0.975)),
        "p_value_one_sided": p_value,
    }])


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving")

robust_family.to_csv(PROC_DIR + "robust_copula_family.csv", index=False)
robust_threshold.to_csv(PROC_DIR + "robust_threshold.csv", index=False)
robust_weights.to_csv(PROC_DIR + "robust_weights.csv", index=False)
robust_perm.to_csv(PROC_DIR + "robust_permutation.csv", index=False)

# Compact summary table for the paper
summary_rows = []
summary_rows.append({"check": "copula_family_amp_Gumbel",
                      "value": float(robust_family.loc[robust_family["family"] == "Gumbel_U", "amp_c_s"].iloc[0])})
summary_rows.append({"check": "copula_family_amp_StudentT",
                      "value": float(robust_family.loc[robust_family["family"] == "StudentT_U(nu=4)", "amp_c_s"].iloc[0])})
summary_rows.append({"check": "copula_family_amp_Clayton_L",
                      "value": float(robust_family.loc[robust_family["family"] == "Clayton_L", "amp_c_s"].iloc[0])})
for _, r in robust_threshold.iterrows():
    summary_rows.append({
        "check": f"DB_crisis_at_q{int(r['threshold_q']*100)}",
        "value": float(r["DB_crisis"]),
    })
for _, r in robust_weights[robust_weights["regime"] == "crisis"].iterrows():
    summary_rows.append({
        "check": f"DB_crisis_weights_{r['weights']}",
        "value": float(r["DB"]),
    })
if "p_value_one_sided" in robust_perm.columns:
    summary_rows.append({"check": "permutation_p_value",
                          "value": float(robust_perm["p_value_one_sided"].iloc[0])})

table_robust = pd.DataFrame(summary_rows)
table_robust.to_csv(RESULTS_DIR + "table_robustness.csv", index=False)

for f in ("robust_copula_family.csv", "robust_threshold.csv",
           "robust_weights.csv", "robust_permutation.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table_robustness.csv")


print("\n" + "=" * 64)
print("DONE — Robustness suite complete")
print("=" * 64)
print("  Headline robustness findings:")
amp_g = robust_family.loc[robust_family["family"] == "Gumbel_U", "amp_c_s"].iloc[0]
amp_t = robust_family.loc[robust_family["family"] == "StudentT_U(nu=4)", "amp_c_s"].iloc[0]
print(f"    λ_U crisis/stable amp — Gumbel: {amp_g:.3f}  Student-t: {amp_t:.3f}")
if not robust_threshold.empty:
    db_range = robust_threshold["DB_crisis"].agg(["min", "max"])
    print(f"    DB_crisis across thresholds: [{db_range['min']:+.4f}, {db_range['max']:+.4f}]")
if "p_value_one_sided" in robust_perm.columns:
    p = robust_perm["p_value_one_sided"].iloc[0]
    print(f"    Permutation p-value (λ_U amp): {p:.4f}")
print(f"\nNext: run 10_rvine_benchmarks.py")
print("=" * 64)
