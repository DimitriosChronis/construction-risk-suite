"""
Paper 5 — R-vine copula (the "vine" claim in the title)
===================================
Fits a proper regular vine copula over the 5 type-level cost-shock return
series using pyvinecopulib. This complements the pairwise-Gumbel λ_U
network of script 03 with a HIERARCHICAL decomposition — the standard
answer to the reviewer question: "you call it a vine but only use
pairwise λ_U; where is the vine?".

Methodology:
    1. Transform each type return series to pseudo-observations (empirical CDF).
    2. Fit an R-vine with family selection (Indep / Gaussian / Student /
       Clayton / Gumbel / Frank / Joe + their rotations) via AIC.
    3. Extract:
         - selected family per edge
         - Kendall's τ per edge
         - Gumbel-equivalent λ_U per edge  (2 − 2^(1−τ), τ>0)
       for every tree level.
    4. Refit on stable vs crisis sub-samples (endogenous regime from 03)
       and compare tree-1 average λ_U.
    5. Report:
         - R-vine structure (matrix) and family matrix
         - Log-likelihood, AIC, BIC vs a Gaussian-only vine benchmark

Inputs:
    portfolio_type_returns_theoretical.csv
    dynamic_copula_regimes.csv

Outputs:
    data/processed/rvine_structure.csv        (tree, edge, family, par, tau, lamU)
    data/processed/rvine_comparison.csv       (all-vine vs stable vs crisis)
    data/processed/rvine_fit_summary.csv      (LL, AIC, BIC vs Gaussian benchmark)
    results/table_rvine.csv                   (paper appendix table)
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import pyvinecopulib as pv
    HAVE_PV = True
except ImportError:
    HAVE_PV = False

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — R-vine copula")
print("=" * 64)

if not HAVE_PV:
    sys.exit("  ✗ pyvinecopulib not available — skip")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading data")

rets = pd.read_csv(PROC_DIR + "portfolio_type_returns_theoretical.csv",
                    index_col=0, parse_dates=True).sort_index()
reg  = pd.read_csv(PROC_DIR + "dynamic_copula_regimes.csv",
                    index_col=0, parse_dates=True).sort_index()

types = list(rets.columns)
K = len(types)
rets = rets.dropna()
reg_al = reg.reindex(rets.index)
print(f"  Types ({K}): {types}")
print(f"  Months: {len(rets)}  ({rets.index.min():%Y-%m} → {rets.index.max():%Y-%m})")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def to_pseudo(u: np.ndarray) -> np.ndarray:
    """Empirical CDF (pseudo-observations) over rows."""
    n, d = u.shape
    r = np.empty_like(u, dtype=float)
    for j in range(d):
        r[:, j] = pd.Series(u[:, j]).rank(method="average").to_numpy()
    return r / (n + 1.0)


def gumbel_lambda_U(tau: float) -> float:
    if np.isnan(tau) or tau <= 0:
        return 0.0
    return 2.0 - 2.0 ** (1.0 - tau)


# pyvinecopulib family codes (pv.BicopFamily enum members)
FAMILY_NAME = {
    pv.BicopFamily.indep:    "Indep",
    pv.BicopFamily.gaussian: "Gaussian",
    pv.BicopFamily.student:  "Student",
    pv.BicopFamily.clayton:  "Clayton",
    pv.BicopFamily.gumbel:   "Gumbel",
    pv.BicopFamily.frank:    "Frank",
    pv.BicopFamily.joe:      "Joe",
    pv.BicopFamily.bb1:      "BB1",
    pv.BicopFamily.bb6:      "BB6",
    pv.BicopFamily.bb7:      "BB7",
    pv.BicopFamily.bb8:      "BB8",
    pv.BicopFamily.tll:      "TLL",
}


def fit_vine(data: np.ndarray, family_set=None, label: str = "") -> dict:
    """Fit an R-vine and return summary dict + per-edge info."""
    u = to_pseudo(data)
    controls = pv.FitControlsVinecop(
        family_set=family_set if family_set is not None else [
            pv.BicopFamily.indep,
            pv.BicopFamily.gaussian,
            pv.BicopFamily.student,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
            pv.BicopFamily.joe,
        ],
        selection_criterion="aic",
        trunc_lvl=K - 1,
    )
    vc = pv.Vinecop.from_data(data=u, controls=controls)

    # Scan all trees / edges
    edges = []
    for tree in range(K - 1):
        for edge in range(K - 1 - tree):
            bicop = vc.get_pair_copula(tree, edge)
            fam = FAMILY_NAME.get(bicop.family, str(bicop.family))
            tau = float(bicop.tau)
            pars = list(np.atleast_1d(bicop.parameters).flatten())
            edges.append({
                "vine":       label,
                "tree":       tree + 1,
                "edge":       edge + 1,
                "family":     fam,
                "rotation":   int(bicop.rotation),
                "tau":        tau,
                "lambda_U":   gumbel_lambda_U(tau),
                "parameters": ",".join(f"{p:.4f}" for p in pars),
            })

    return {
        "vine":    vc,
        "loglik":  float(vc.loglik(u)),
        "aic":     float(vc.aic(u)),
        "bic":     float(vc.bic(u)),
        "npars":   int(vc.npars),
        "edges":   edges,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. FULL-SAMPLE R-VINE (all families)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Fitting full-sample R-vine (all families, AIC)")
data = rets[types].to_numpy()
full_fit = fit_vine(data, label="full")
print(f"  log-lik = {full_fit['loglik']:.2f}  AIC = {full_fit['aic']:.2f}  "
      f"BIC = {full_fit['bic']:.2f}  npars = {full_fit['npars']}")

print("\n  Tree 1 edges (unconditional pair copulas):")
for e in full_fit["edges"]:
    if e["tree"] == 1:
        print(f"    edge {e['edge']}: {e['family']:8s} rot={e['rotation']:3d}  "
              f"τ={e['tau']:+.3f}  λ_U={e['lambda_U']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GAUSSIAN-ONLY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Fitting Gaussian-only benchmark vine")
gauss_fit = fit_vine(data,
                      family_set=[pv.BicopFamily.gaussian],
                      label="gaussian_bench")
print(f"  log-lik = {gauss_fit['loglik']:.2f}  AIC = {gauss_fit['aic']:.2f}  "
      f"BIC = {gauss_fit['bic']:.2f}  npars = {gauss_fit['npars']}")

delta_ll  = full_fit["loglik"] - gauss_fit["loglik"]
delta_aic = full_fit["aic"]    - gauss_fit["aic"]
delta_bic = full_fit["bic"]    - gauss_fit["bic"]
print(f"  R-vine vs Gaussian:  ΔLL = {delta_ll:+.2f}  "
      f"ΔAIC = {delta_aic:+.2f}  ΔBIC = {delta_bic:+.2f}")
if delta_aic < 0:
    print("  → R-vine preferred by AIC (richer tail dependence)")
else:
    print("  → Gaussian suffices — no evidence of non-Gaussian tails")


# ══════════════════════════════════════════════════════════════════════════════
# 3. STABLE vs CRISIS SUB-SAMPLE VINES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Fitting stable vs crisis sub-sample R-vines")

regime_series = reg_al["regime"]
stable_data = rets.loc[regime_series == "stable", types].to_numpy()
crisis_data = rets.loc[regime_series == "crisis", types].to_numpy()
print(f"  stable n={len(stable_data)}  crisis n={len(crisis_data)}")

cmp_rows = [{
    "sample":  "full",
    "n":       len(data),
    "loglik":  full_fit["loglik"],
    "aic":     full_fit["aic"],
    "bic":     full_fit["bic"],
    "tree1_avg_lambdaU": float(np.mean(
        [e["lambda_U"] for e in full_fit["edges"] if e["tree"] == 1])),
}]

for label, sub in (("stable", stable_data), ("crisis", crisis_data)):
    if len(sub) < K * 4:
        print(f"  {label}: too few rows ({len(sub)}) — skipped")
        continue
    fit = fit_vine(sub, label=label)
    tree1_lamU = [e["lambda_U"] for e in fit["edges"] if e["tree"] == 1]
    cmp_rows.append({
        "sample":  label,
        "n":       len(sub),
        "loglik":  fit["loglik"],
        "aic":     fit["aic"],
        "bic":     fit["bic"],
        "tree1_avg_lambdaU": float(np.mean(tree1_lamU)),
    })
    # Append sub-sample edges to full edges list
    full_fit["edges"].extend(fit["edges"])

rvine_cmp = pd.DataFrame(cmp_rows)
print("\n  R-vine regime comparison:")
print(rvine_cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Amplification test from R-vine tree-1 λ_U
if {"stable", "crisis"}.issubset(set(rvine_cmp["sample"])):
    amp = (rvine_cmp.loc[rvine_cmp["sample"] == "crisis", "tree1_avg_lambdaU"].iloc[0] /
           rvine_cmp.loc[rvine_cmp["sample"] == "stable", "tree1_avg_lambdaU"].iloc[0])
    print(f"\n  R-vine tree-1 λ_U crisis/stable amplification: {amp:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving")

rvine_edges = pd.DataFrame(full_fit["edges"])
rvine_edges.to_csv(PROC_DIR + "rvine_structure.csv", index=False)
rvine_cmp.to_csv(PROC_DIR + "rvine_comparison.csv", index=False)

fit_summary = pd.DataFrame([
    {"model": "R-vine (all families)",
     "loglik": full_fit["loglik"], "aic": full_fit["aic"],
     "bic": full_fit["bic"], "npars": full_fit["npars"]},
    {"model": "Vine (Gaussian only)",
     "loglik": gauss_fit["loglik"], "aic": gauss_fit["aic"],
     "bic": gauss_fit["bic"], "npars": gauss_fit["npars"]},
    {"model": "Δ (R-vine − Gaussian)",
     "loglik": delta_ll, "aic": delta_aic,
     "bic": delta_bic, "npars": full_fit["npars"] - gauss_fit["npars"]},
])
fit_summary.to_csv(PROC_DIR + "rvine_fit_summary.csv", index=False)

# Paper appendix table: tree-1 edges with families
tree1 = rvine_edges[(rvine_edges["tree"] == 1) &
                     (rvine_edges["vine"] == "full")].copy()
tree1 = tree1[["edge", "family", "rotation", "tau", "lambda_U", "parameters"]]
tree1.to_csv(RESULTS_DIR + "table_rvine.csv", index=False)

for f in ("rvine_structure.csv", "rvine_comparison.csv", "rvine_fit_summary.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table_rvine.csv")

print("\n" + "=" * 64)
print("DONE — R-vine complete")
print("=" * 64)
print(f"  R-vine vs Gaussian ΔAIC: {delta_aic:+.2f}  "
      f"(negative ⇒ R-vine preferred)")
print(f"\nNext: run 11_benchmarks.py")
print("=" * 64)
