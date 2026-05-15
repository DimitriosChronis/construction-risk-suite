"""
Paper 5 — Construction Systemic Risk Index (CSRI)
===================================
Final methodological component + AiC (Automation-in-Construction) angle.

Builds a MONTHLY COMPOSITE INDEX that summarises portfolio-wide tail risk
in a single scalar intended to be re-computed automatically each month:

    CSRI(t) = mean over components of their 60-month rolling z-score

Components (all already produced by 03-06):
    1. network_avg_lambdaU     (03)  — avg upper tail dependence across pairs
    2. mean_CI                 (04)  — network density of contagion links
    3. max_CI_type             (04)  — worst concentration in any single type
    4. (1 - div_benefit)       (05)  — loss of diversification benefit (rolling)
    5. agent_prob_crisis       (06)  — LSTM portfolio agent P(crisis) (if available)

Granger causality tests:  CSRI(t-k)  →  extreme_i(t)    for k = 1, 2, 3
    extreme_i(t) = 1 if type i monthly return falls below its 10th percentile.

Block bootstrap CIs (1 000 resamples, block=12 months) on:
    - mean λ_U (full / stable / crisis)
    - mean CI  (full / stable / crisis)
    - crisis amplification (crisis / stable)
    - diversification benefit (full / stable / crisis)

Inputs  (all under data/processed/):
    dynamic_copula_network_avg.csv        (from 03)
    dynamic_copula_regimes.csv            (from 03)
    contagion_index.csv                   (from 04)
    portfolio_es_rolling.csv              (from 05)
    agent_predictions.csv                 (from 06, optional)
    portfolio_type_returns_theoretical.csv (from 02)

Outputs:
    data/processed/csri_monthly.csv       (date, CSRI + components + regime)
    data/processed/granger_results.csv
    data/processed/bootstrap_ci.csv
    results/table6_csri.csv               (paper Table 6)
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAVE_STATS = True
except ImportError:
    HAVE_STATS = False

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR     = "../data/processed/"
RESULTS_DIR  = "../results/"
ZSCORE_WIN   = 60          # rolling window for z-scoring components
GRANGER_LAGS = [1, 2, 3]
N_BOOTSTRAP  = 1000
BLOCK_SIZE   = 12          # block bootstrap block length
EXTREME_Q    = 0.10        # lower tail threshold for "extreme" month
RNG          = np.random.default_rng(42)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("Paper 5 — CSRI (Construction Systemic Risk Index)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL INGREDIENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading CSRI ingredients")


def load_csv(name: str, required: bool = True) -> pd.DataFrame | None:
    p = os.path.join(PROC_DIR, name)
    if not os.path.exists(p):
        if required:
            sys.exit(f"  ✗ Missing {p}")
        print(f"  (optional) skip — {name}")
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()


network  = load_csv("dynamic_copula_network_avg.csv")
regimes  = load_csv("dynamic_copula_regimes.csv")
CI       = load_csv("contagion_index.csv")
es_roll  = load_csv("portfolio_es_rolling.csv")
agent    = load_csv("agent_predictions.csv", required=False)
type_ret = load_csv("portfolio_type_returns_theoretical.csv")

print(f"  network:   {network.shape}")
print(f"  regimes:   {regimes.shape}")
print(f"  CI:        {CI.shape}")
print(f"  es_roll:   {es_roll.shape}")
print(f"  agent:     {agent.shape if agent is not None else '(missing)'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ASSEMBLE COMPONENTS + ROLLING Z-SCORE → CSRI
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 2: Building CSRI components (rolling z-score, window={ZSCORE_WIN}M)")

comp = pd.DataFrame(index=network.index)
comp["lambdaU"]      = network["avg_lambdaU"]
comp["mean_CI"]      = CI.mean(axis=1)
comp["max_CI"]       = CI.max(axis=1)
comp["loss_div_ben"] = 1.0 - es_roll["div_benefit"].reindex(network.index)

if agent is not None:
    # Use best single-model probability if available (prefer lstm → xgb → logit)
    for col in ("lstm", "xgb", "logit"):
        if col in agent.columns and agent[col].notna().sum() > 20:
            comp["agent_prob"] = agent[col].reindex(network.index)
            print(f"  Using agent column: {col}")
            break

comp = comp.dropna(axis=1, how="all")
print(f"  Components: {list(comp.columns)}")


def rolling_z(s: pd.Series, win: int) -> pd.Series:
    mu    = s.rolling(win, min_periods=max(12, win // 2)).mean()
    sigma = s.rolling(win, min_periods=max(12, win // 2)).std(ddof=0)
    return (s - mu) / sigma.replace(0, np.nan)


comp_z = comp.apply(lambda col: rolling_z(col, ZSCORE_WIN))
comp_z.columns = [f"z_{c}" for c in comp_z.columns]

csri = pd.DataFrame(index=comp.index)
csri["CSRI"] = comp_z.mean(axis=1, skipna=True)
csri = csri.join(comp).join(comp_z)
csri = csri.join(regimes[["regime", "regime_exo"]], how="left")
csri.index.name = "date"

n_valid = csri["CSRI"].notna().sum()
print(f"  CSRI computed for {n_valid} months ("
      f"{csri['CSRI'].dropna().index.min():%Y-%m} → "
      f"{csri['CSRI'].dropna().index.max():%Y-%m})")
print(f"  Mean CSRI — stable: "
      f"{csri.loc[csri['regime']=='stable','CSRI'].mean():+.3f}  |  "
      f"crisis: {csri.loc[csri['regime']=='crisis','CSRI'].mean():+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRANGER CAUSALITY:  CSRI(t-k)  →  extreme_i(t)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 3: Granger causality tests  (lags={GRANGER_LAGS})")

types = list(type_ret.columns)
granger_rows = []

if HAVE_STATS:
    csri_series = csri["CSRI"].dropna()
    for t in types:
        rt = type_ret[t].reindex(csri_series.index).dropna()
        thr = rt.quantile(EXTREME_Q)
        extreme = (rt < thr).astype(float)
        common = extreme.index.intersection(csri_series.index)
        if len(common) < 60:
            continue
        paired = pd.DataFrame({
            "extreme": extreme.reindex(common).values,
            "csri":    csri_series.reindex(common).values,
        }).dropna()
        # statsmodels expects: dep in col 0, causal candidate in col 1
        try:
            res = grangercausalitytests(
                paired[["extreme", "csri"]], maxlag=max(GRANGER_LAGS),
                verbose=False,
            )
            for k in GRANGER_LAGS:
                f_stat, p_val = res[k][0]["ssr_ftest"][0], res[k][0]["ssr_ftest"][1]
                granger_rows.append({
                    "type":      t,
                    "lag":       k,
                    "F_stat":    f_stat,
                    "p_value":   p_val,
                    "reject_05": bool(p_val < 0.05),
                    "n_obs":     int(len(paired)),
                })
        except Exception as e:
            print(f"  ! Granger failed for {t}: {e}")
else:
    print("  (statsmodels unavailable — Granger tests skipped)")

granger = pd.DataFrame(granger_rows)
if not granger.empty:
    print(granger.to_string(index=False,
                             float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 4. BLOCK BOOTSTRAP CIs
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 4: Block bootstrap  (B={N_BOOTSTRAP}, block={BLOCK_SIZE})")


def block_bootstrap_indices(n: int, block: int, rng) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def stat_fn(df: pd.DataFrame) -> dict:
    # df has: lambdaU, mean_CI, div_benefit, regime
    stable = df[df["regime"] == "stable"]
    crisis = df[df["regime"] == "crisis"]
    return {
        "lambdaU_full":      df["lambdaU"].mean(),
        "lambdaU_stable":    stable["lambdaU"].mean() if len(stable) else np.nan,
        "lambdaU_crisis":    crisis["lambdaU"].mean() if len(crisis) else np.nan,
        "lambdaU_amp":       (crisis["lambdaU"].mean() / stable["lambdaU"].mean())
                              if len(stable) and len(crisis) else np.nan,
        "mean_CI_full":      df["mean_CI"].mean(),
        "mean_CI_stable":    stable["mean_CI"].mean() if len(stable) else np.nan,
        "mean_CI_crisis":    crisis["mean_CI"].mean() if len(crisis) else np.nan,
        "div_benefit_full":  df["div_benefit"].mean(),
        "div_benefit_stable": stable["div_benefit"].mean() if len(stable) else np.nan,
        "div_benefit_crisis": crisis["div_benefit"].mean() if len(crisis) else np.nan,
    }


boot_df = pd.DataFrame({
    "lambdaU":     network["avg_lambdaU"],
    "mean_CI":     CI.mean(axis=1),
    "div_benefit": es_roll["div_benefit"].reindex(network.index),
    "regime":      regimes["regime"].reindex(network.index),
}).dropna()
print(f"  Bootstrap sample size: {len(boot_df)}")

point = stat_fn(boot_df)
boot_stats = {k: [] for k in point}
arr = boot_df.reset_index(drop=True)
n = len(arr)

for b in range(N_BOOTSTRAP):
    idx = block_bootstrap_indices(n, BLOCK_SIZE, RNG)
    samp = arr.iloc[idx]
    s = stat_fn(samp)
    for k, v in s.items():
        boot_stats[k].append(v)

boot_rows = []
for k, vals in boot_stats.items():
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        continue
    boot_rows.append({
        "statistic": k,
        "point":     point[k],
        "mean_boot": float(np.mean(v)),
        "se_boot":   float(np.std(v, ddof=1)),
        "ci_low":    float(np.quantile(v, 0.025)),
        "ci_high":   float(np.quantile(v, 0.975)),
        "n_boot":    int(len(v)),
    })
boot_tab = pd.DataFrame(boot_rows)
print(boot_tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 5. TABLE 6  (CSRI summary per regime + headline bootstrap figures)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Building Table 6")

table6_rows = []
for reg in ("full", "stable", "crisis"):
    if reg == "full":
        sub = csri
    else:
        sub = csri[csri["regime"] == reg]
    if len(sub) < 5:
        continue
    table6_rows.append({
        "regime":        reg,
        "n_months":      int(len(sub)),
        "mean_CSRI":     sub["CSRI"].mean(),
        "std_CSRI":      sub["CSRI"].std(),
        "pct_CSRI>1":    float((sub["CSRI"] > 1).mean()),
        "mean_lambdaU":  sub["lambdaU"].mean(),
        "mean_CI":       sub["mean_CI"].mean(),
    })
table6 = pd.DataFrame(table6_rows)
print(table6.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Saving")

csri.to_csv(os.path.join(PROC_DIR, "csri_monthly.csv"))
granger.to_csv(os.path.join(PROC_DIR, "granger_results.csv"), index=False)
boot_tab.to_csv(os.path.join(PROC_DIR, "bootstrap_ci.csv"), index=False)
table6.to_csv(os.path.join(RESULTS_DIR, "table6_csri.csv"), index=False)

for f in ("csri_monthly.csv", "granger_results.csv", "bootstrap_ci.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table6_csri.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — CSRI computed, Granger tested, CIs bootstrapped")
print("=" * 64)

if not granger.empty:
    sig = granger[granger["reject_05"]]
    print(f"  Granger CSRI→extreme: {len(sig)}/{len(granger)} tests significant @ 5%")
    if len(sig) > 0:
        print("    Significant (type, lag, p):")
        for _, r in sig.iterrows():
            print(f"      {r['type']:9s}  lag={int(r['lag'])}  p={r['p_value']:.4f}")

amp_row = boot_tab[boot_tab["statistic"] == "lambdaU_amp"]
if not amp_row.empty:
    r = amp_row.iloc[0]
    print(f"  λ_U crisis/stable amplification: {r['point']:.3f} "
          f"[95% CI: {r['ci_low']:.3f}, {r['ci_high']:.3f}]")

print(f"\nNext: run 08_publication_figures.py")
print("=" * 64)
