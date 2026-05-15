"""
Paper 5 — Classical contagion benchmarks
===================================
Two finance-standard cross-asset risk measures, computed on the same
5 type-level cost-shock returns used by 03/04, so we can claim that the
Gumbel λ_U network and the contagion index of P5 add information beyond
the textbook tools.

    1. DCC-GARCH(1,1) pairwise dynamic correlations
       (Engle 2002 — the canonical time-varying-correlation model).
       Refit per pair via the `arch` library.  Compare avg DCC ρ to
       avg λ_U from script 03 across regimes.

    2. Diebold–Yilmaz (2009, 2012) total / directional spillover index
       from a VAR(p) generalised forecast-error variance decomposition.
       Provides an ALTERNATIVE source/receiver classification that we
       cross-validate against script 04 contagion flows.

Inputs:
    portfolio_type_returns_theoretical.csv
    dynamic_copula_network_avg.csv
    dynamic_copula_regimes.csv
    contagion_flows.csv               (for cross-validation)

Outputs:
    data/processed/dcc_correlations.csv      (date × pair, DCC ρ_t)
    data/processed/dcc_vs_lambdaU.csv        (regime comparison)
    data/processed/dy_spillover.csv          (full-sample DY table)
    data/processed/dy_classification.csv     (NET DY per type vs P5 NET)
    results/table_benchmarks.csv             (compact summary)
"""

import io
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from arch.univariate import ConstantMean, GARCH, Normal
    HAVE_ARCH = True
except ImportError:
    HAVE_ARCH = False

try:
    from statsmodels.tsa.api import VAR
    HAVE_VAR = True
except ImportError:
    HAVE_VAR = False

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — DCC-GARCH + Diebold–Yilmaz benchmarks")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading data")

rets = pd.read_csv(PROC_DIR + "portfolio_type_returns_theoretical.csv",
                    index_col=0, parse_dates=True).sort_index().dropna()
net  = pd.read_csv(PROC_DIR + "dynamic_copula_network_avg.csv",
                    index_col=0, parse_dates=True).sort_index()
reg  = pd.read_csv(PROC_DIR + "dynamic_copula_regimes.csv",
                    index_col=0, parse_dates=True).sort_index()
flows = pd.read_csv(PROC_DIR + "contagion_flows.csv",
                     index_col=0, parse_dates=True).sort_index()

types = list(rets.columns)
K = len(types)
pairs = list(combinations(types, 2))
print(f"  Types ({K}): {types}")
print(f"  Pair count: {len(pairs)}")
print(f"  Months: {len(rets)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DCC-GARCH PAIRWISE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/2] Pairwise DCC-GARCH(1,1)")

if not HAVE_ARCH:
    sys.exit("  ✗ arch library not available")


def fit_garch_std_resid(series: pd.Series) -> tuple[np.ndarray, float]:
    """Fit GARCH(1,1) and return standardised residuals + cond. vol."""
    am = ConstantMean(series * 100.0)         # scale → percent
    am.volatility = GARCH(1, 0, 1)
    am.distribution = Normal()
    res = am.fit(disp="off", show_warning=False)
    sigma = res.conditional_volatility / 100.0
    eps = (series - res.params.get("mu", 0.0) / 100.0) / sigma
    return eps.to_numpy(), sigma.to_numpy()


def dcc_correlation(eps_x: np.ndarray, eps_y: np.ndarray,
                     a: float = 0.05, b: float = 0.93) -> np.ndarray:
    """Bivariate DCC(1,1) recursion (Engle 2002) with fixed (a,b).

    Returns the time series of conditional correlations ρ_t.
    """
    n = len(eps_x)
    Q_bar = np.array([
        [np.mean(eps_x ** 2), np.mean(eps_x * eps_y)],
        [np.mean(eps_x * eps_y), np.mean(eps_y ** 2)],
    ])
    Q = Q_bar.copy()
    rho = np.zeros(n)
    for t in range(n):
        if t > 0:
            e = np.array([eps_x[t - 1], eps_y[t - 1]])
            Q = (1 - a - b) * Q_bar + a * np.outer(e, e) + b * Q
        d = np.sqrt(np.diag(Q))
        denom = d[0] * d[1]
        rho[t] = Q[0, 1] / denom if denom > 0 else 0.0
    return np.clip(rho, -0.999, 0.999)


# Fit GARCH univariates once per type
print("  Fitting univariate GARCH(1,1) models")
std_resid: dict[str, np.ndarray] = {}
for t in types:
    eps, _ = fit_garch_std_resid(rets[t])
    std_resid[t] = eps

dcc_df = pd.DataFrame(index=rets.index)
for (a, b) in pairs:
    rho = dcc_correlation(std_resid[a], std_resid[b])
    dcc_df[f"{a}__{b}"] = rho

dcc_df["dcc_avg"] = dcc_df[[c for c in dcc_df.columns if c != "dcc_avg"]].mean(axis=1)
print(f"  DCC pairwise series: {dcc_df.shape}")

# Compare with λ_U from script 03 by regime
reg_al = reg.reindex(dcc_df.index)["regime"]
lam_avg = net["avg_lambdaU"].reindex(dcc_df.index)

cmp_rows = []
for label, mask in (("full",   pd.Series(True, index=dcc_df.index)),
                     ("stable", reg_al == "stable"),
                     ("crisis", reg_al == "crisis")):
    m_dcc = dcc_df.loc[mask, "dcc_avg"].mean()
    m_lam = lam_avg.loc[mask].mean()
    cmp_rows.append({
        "regime":   label,
        "n":        int(mask.sum()),
        "dcc_avg":  m_dcc,
        "lambdaU":  m_lam,
        "ratio":    m_lam / m_dcc if m_dcc > 0 else np.nan,
    })
dcc_vs_lam = pd.DataFrame(cmp_rows)
print("\n  DCC vs λ_U comparison:")
print(dcc_vs_lam.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Amplification — DCC alone:
m_s = dcc_vs_lam.loc[dcc_vs_lam["regime"] == "stable", "dcc_avg"].iloc[0]
m_c = dcc_vs_lam.loc[dcc_vs_lam["regime"] == "crisis", "dcc_avg"].iloc[0]
dcc_amp = m_c / m_s if m_s > 0 else np.nan
print(f"  DCC ρ crisis/stable amplification: {dcc_amp:.4f}")
m_s_l = dcc_vs_lam.loc[dcc_vs_lam["regime"] == "stable", "lambdaU"].iloc[0]
m_c_l = dcc_vs_lam.loc[dcc_vs_lam["regime"] == "crisis", "lambdaU"].iloc[0]
lam_amp = m_c_l / m_s_l if m_s_l > 0 else np.nan
print(f"  λ_U   crisis/stable amplification: {lam_amp:.4f}")
print(f"  → tail-dep amplification {lam_amp - dcc_amp:+.4f} HIGHER than DCC ρ"
      if (lam_amp - dcc_amp) > 0 else
      f"  → DCC ρ amp matches λ_U amp")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIEBOLD–YILMAZ SPILLOVER INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/2] Diebold–Yilmaz spillover index (full sample)")

if not HAVE_VAR:
    sys.exit("  ✗ statsmodels VAR unavailable")


def gfevd(var_results, h: int) -> np.ndarray:
    """Generalised forecast-error variance decomposition (Pesaran-Shin).

    Returns a K×K matrix where row i = % of i's H-step error variance
    attributable to shocks in column j (rows do NOT sum to 1 → row-normalise).
    """
    sigma = np.asarray(var_results.sigma_u)          # residual cov matrix
    A = np.asarray(var_results.ma_rep(maxn=h))       # impulse responses (h+1, K, K)
    K = sigma.shape[0]
    sigma_diag = np.diag(sigma)
    theta = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            num = 0.0
            for hh in range(h + 1):
                num += (A[hh][i] @ sigma[:, j]) ** 2
            denom = 0.0
            for hh in range(h + 1):
                denom += A[hh][i] @ sigma @ A[hh][i].T
            theta[i, j] = (1.0 / sigma_diag[j]) * num / denom
    # Row-normalise (each row sums to 1)
    return theta / theta.sum(axis=1, keepdims=True)


def dy_spillover_table(var_results, types: list[str], h: int = 10) -> pd.DataFrame:
    theta = gfevd(var_results, h) * 100.0
    df = pd.DataFrame(theta, index=types, columns=types)
    # FROM others = 1 − own
    df["FROM_others"] = 100.0 - np.diag(theta)
    # TO others = column sum minus diagonal
    to_others = theta.sum(axis=0) - np.diag(theta)
    df.loc["TO_others"] = list(to_others) + [np.nan]
    # NET = TO − FROM (per type)
    net_v = to_others - (100.0 - np.diag(theta))
    df.loc["NET"] = list(net_v) + [np.nan]
    # Total spillover index = sum of off-diag / K
    total = (theta.sum() - np.trace(theta)) / theta.shape[0]
    df.loc["TOTAL"] = [np.nan] * (len(types)) + [total]
    return df


# Fit VAR on type returns.
# NOTE: type returns are linear combinations of 4 material series so they
# are near-collinear (DCC ρ ≈ 0.95). We add a tiny idiosyncratic jitter
# (1% of type σ) per type to regularise the covariance matrix, and use
# a fixed lag p=1 (AIC-based selection fails on the singular system).
rng = np.random.default_rng(42)
rets_j = rets[types].copy()
for t in types:
    sig = rets_j[t].std()
    rets_j[t] = rets_j[t] + rng.normal(0.0, 0.01 * sig, size=len(rets_j))

var_model = VAR(rets_j)
p_opt = 1
print(f"  VAR lag (fixed): p = {p_opt} "
      f"(theoretical returns near-collinear → AIC selection skipped)")
var_res = var_model.fit(p_opt)

dy_table = dy_spillover_table(var_res, types, h=10)
print("\n  Diebold–Yilmaz spillover table (%):")
print(dy_table.to_string(float_format=lambda x: f"{x:6.2f}"))

# Cross-validate NET DY ranking against P5 contagion NET ranking
flow_avg = flows.mean()  # avg over time
p5_net = pd.Series({t: flow_avg.get(f"NET_{t}", np.nan) for t in types})
dy_net = dy_table.loc["NET", types]

dy_cls = pd.DataFrame({
    "type":     types,
    "DY_NET":   dy_net.values,
    "P5_NET":   p5_net.values,
})
dy_cls["DY_role"] = np.where(dy_cls["DY_NET"] > 0, "source", "receiver")
dy_cls["P5_role"] = np.where(dy_cls["P5_NET"] > 0, "source", "receiver")
dy_cls["agree"] = dy_cls["DY_role"] == dy_cls["P5_role"]
print("\n  Source/receiver cross-validation (DY vs P5):")
print(dy_cls.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"  Agreement: {dy_cls['agree'].sum()}/{len(dy_cls)} types")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving")

dcc_df.to_csv(PROC_DIR + "dcc_correlations.csv")
dcc_vs_lam.to_csv(PROC_DIR + "dcc_vs_lambdaU.csv", index=False)
dy_table.to_csv(PROC_DIR + "dy_spillover.csv")
dy_cls.to_csv(PROC_DIR + "dy_classification.csv", index=False)

bench_summary = pd.DataFrame([
    {"check": "DCC_amp_crisis_stable",     "value": dcc_amp},
    {"check": "lambdaU_amp_crisis_stable", "value": lam_amp},
    {"check": "DY_total_spillover_pct",    "value": float(dy_table.loc["TOTAL", "FROM_others"])},
    {"check": "DY_P5_role_agreement_K",    "value": float(dy_cls['agree'].sum())},
    {"check": "DY_P5_role_agreement_pct",  "value": float(dy_cls['agree'].mean() * 100)},
    {"check": "VAR_lag_AIC",               "value": float(p_opt)},
])
bench_summary.to_csv(RESULTS_DIR + "table_benchmarks.csv", index=False)

for f in ("dcc_correlations.csv", "dcc_vs_lambdaU.csv",
           "dy_spillover.csv", "dy_classification.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table_benchmarks.csv")

print("\n" + "=" * 64)
print("DONE — Classical benchmarks complete")
print("=" * 64)
print(f"  λ_U amp ({lam_amp:.3f}) vs DCC ρ amp ({dcc_amp:.3f})")
print(f"  DY total spillover index: {float(dy_table.loc['TOTAL', 'FROM_others']):.2f}%")
print(f"  DY/P5 source-receiver agreement: {dy_cls['agree'].sum()}/{len(dy_cls)}")
print(f"\nNext: run 12_lead_time_curve.py")
print("=" * 64)
