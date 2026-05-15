"""
Paper 5 — Portfolio Expected Shortfall
===================================
Third methodological + empirical contribution of P5.

Tests the central empirical claim of the paper:

    During crisis regimes, Portfolio ES approaches (or exceeds) the sum of
    individual project ES — classical subadditivity-based diversification
    benefit shrinks or disappears.

Definitions (loss convention: positive number = loss of budget margin):
    r_i(t)  = cost-shock return of project type i at month t (NEGATIVE = loss)
    w_i     = portfolio weight of type i  (equal or budget-share)
    ES_i(α) = −E[r_i | r_i ≤ VaR_i(α)]                 ≥ 0
    ES_Σ    = Σ_i w_i · ES_i(α)                        (weighted sum)
    r_P(t)  = Σ_i w_i · r_i(t)
    ES_P(α) = −E[r_P | r_P ≤ VaR_P(α)]                 ≥ 0

    Diversification benefit:   DB = 1 − ES_P / ES_Σ
        DB > 0  ⇒ classical diversification works
        DB ≈ 0  ⇒ benefit gone
        DB < 0  ⇒ SUPER-additive (crisis amplification)  ← Q1 finding

Per-project contingency buffer:
    contingency_pct_i = |ES_crisis(type_i)| · √(duration_i / 12)

Inputs:
  - data/processed/portfolio_type_returns_theoretical.csv  (from 02)
  - data/processed/portfolio_metadata.csv                  (from 02)
  - data/processed/dynamic_copula_regimes.csv              (from 03)

Outputs:
  - data/processed/portfolio_es_by_regime.csv      (Table 4 body)
  - data/processed/portfolio_es_rolling.csv        (rolling 24M ES_P / ES_Σ)
  - data/processed/project_contingency.csv         (per-project buffer %)
  - results/table4_portfolio_es.csv                (paper Table 4)

Run from: construction-risk-suite/paper5-portfolio-contagion/src/
"""

import io
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
ALPHA       = 0.95           # ES confidence level (worst 5%)
WINDOW      = 24             # rolling window (months)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print(f"Paper 5 — Portfolio ES  (α={ALPHA})")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading inputs")

rets_path    = os.path.join(PROC_DIR, "portfolio_type_returns_theoretical.csv")
meta_path    = os.path.join(PROC_DIR, "portfolio_metadata.csv")
regimes_path = os.path.join(PROC_DIR, "dynamic_copula_regimes.csv")

for p in (rets_path, meta_path, regimes_path):
    if not os.path.exists(p):
        sys.exit(f"  ✗ Missing {p} — run 02 and 03 first")

rets    = pd.read_csv(rets_path, index_col=0, parse_dates=True).sort_index()
meta    = pd.read_csv(meta_path, encoding="utf-8-sig")
regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True).sort_index()

types = list(rets.columns)
K = len(types)
print(f"  Types: {types}")
print(f"  Returns: {len(rets)} months  ({rets.index.min():%Y-%m} → {rets.index.max():%Y-%m})")
print(f"  Projects in metadata: {len(meta)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Computing portfolio weights")

# Equal weights (fallback, always defined)
w_equal = np.full(K, 1.0 / K)

# Budget-share weights from downloaded projects
type_budget = meta.groupby("type")["budget_eur"].sum()
type_budget = type_budget.reindex(types).fillna(0)
if type_budget.sum() == 0:
    w_budget = w_equal.copy()
    print("  ⚠ No budget data — falling back to equal weights")
else:
    w_budget = (type_budget / type_budget.sum()).to_numpy()

print("  Weights:")
for t, we, wb in zip(types, w_equal, w_budget):
    print(f"    {t:9s}: equal={we:.3f}  budget={wb:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ES HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    """ES of a return series at level α. Returns a POSITIVE loss magnitude.

    Uses the left tail of `losses` (since returns are negative = losses).
    """
    x = np.asarray(losses, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan
    var_q = np.quantile(x, 1 - alpha)                  # e.g. 5th percentile
    tail = x[x <= var_q]
    return float(-tail.mean()) if len(tail) else np.nan


def es_suite(window: pd.DataFrame, weights: np.ndarray, alpha: float) -> dict:
    """Compute ES_i per type, ES_Σ (weighted sum), and ES_P (portfolio)."""
    es_per_type = np.array([expected_shortfall(window[t].to_numpy(), alpha)
                             for t in window.columns])
    es_sum  = float(np.nansum(weights * es_per_type))
    r_p     = window.to_numpy() @ weights
    es_port = expected_shortfall(r_p, alpha)
    db      = 1.0 - (es_port / es_sum) if es_sum and es_sum > 0 else np.nan
    return {
        "es_per_type": es_per_type,
        "es_sum":      es_sum,
        "es_port":     es_port,
        "ratio":       es_port / es_sum if es_sum and es_sum > 0 else np.nan,
        "db":          db,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGIME-BY-REGIME ES (Table 4)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 3: Computing ES by regime  (α={ALPHA})")

regimes_al = regimes.reindex(rets.index).dropna()
rets_al = rets.loc[regimes_al.index]

masks = {
    "full":   pd.Series(True, index=rets_al.index),
    "stable": regimes_al["regime"] == "stable",
    "crisis": regimes_al["regime"] == "crisis",
}

table4_rows = []
for w_label, w in (("equal", w_equal), ("budget", w_budget)):
    for reg_label, mask in masks.items():
        sub = rets_al.loc[mask]
        if len(sub) < 5:
            print(f"  {w_label}/{reg_label}: only {len(sub)} months — skipped")
            continue
        res = es_suite(sub, w, ALPHA)
        table4_rows.append({
            "weights":       w_label,
            "regime":        reg_label,
            "n_months":      int(len(sub)),
            "ES_sum":        res["es_sum"],
            "ES_portfolio":  res["es_port"],
            "ratio":         res["ratio"],
            "div_benefit":   res["db"],
        })
        for t, es_i in zip(types, res["es_per_type"]):
            table4_rows[-1][f"ES_{t}"] = es_i

table4 = pd.DataFrame(table4_rows)
cols_lead = ["weights", "regime", "n_months",
             "ES_sum", "ES_portfolio", "ratio", "div_benefit"]
table4 = table4[cols_lead + [f"ES_{t}" for t in types]]
print(table4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROLLING 24M PORTFOLIO ES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 4: Rolling {WINDOW}-month ES_P and ES_Σ (budget-weighted)")

rolling_rows = []
for end in range(WINDOW - 1, len(rets_al)):
    win = rets_al.iloc[end - WINDOW + 1: end + 1]
    res = es_suite(win, w_budget, ALPHA)
    rolling_rows.append({
        "date":         rets_al.index[end],
        "ES_sum":       res["es_sum"],
        "ES_portfolio": res["es_port"],
        "ratio":        res["ratio"],
        "div_benefit":  res["db"],
    })

rolling = pd.DataFrame(rolling_rows).set_index("date")
rolling = rolling.join(regimes_al[["regime"]], how="left")

print(f"  Rolling snapshots: {len(rolling)}")
print(f"  Mean ES_P / ES_Σ ratio — stable: "
      f"{rolling.loc[rolling['regime']=='stable','ratio'].mean():.3f} | "
      f"crisis: {rolling.loc[rolling['regime']=='crisis','ratio'].mean():.3f}")
print(f"  Mean div. benefit     — stable: "
      f"{rolling.loc[rolling['regime']=='stable','div_benefit'].mean():.3f} | "
      f"crisis: {rolling.loc[rolling['regime']=='crisis','div_benefit'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PER-PROJECT CONTINGENCY BUFFER
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Computing per-project contingency buffers")

# Crisis ES per type, from budget-weighted regime breakdown
crisis_row = table4[
    (table4["weights"] == "budget") & (table4["regime"] == "crisis")
]
if crisis_row.empty:
    print("  ⚠ No crisis regime months — using 'full' ES instead")
    crisis_row = table4[
        (table4["weights"] == "budget") & (table4["regime"] == "full")
    ]
crisis_es_by_type = {t: float(crisis_row[f"ES_{t}"].iloc[0]) for t in types}

contingency = meta.copy()
contingency["es_crisis_type"] = contingency["type"].map(crisis_es_by_type)
contingency["contingency_pct"] = (
    contingency["es_crisis_type"] * np.sqrt(contingency["duration_m"] / 12.0)
)
contingency["contingency_eur"] = (
    contingency["budget_eur"] * contingency["contingency_pct"]
)

by_type = contingency.groupby("type").agg(
    n=("project_id", "size"),
    mean_contingency_pct=("contingency_pct", "mean"),
    total_contingency_eur=("contingency_eur", "sum"),
)
print(by_type.to_string(float_format=lambda x: f"{x:,.3f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Saving")

table4.to_csv(os.path.join(PROC_DIR, "portfolio_es_by_regime.csv"), index=False)
table4.to_csv(os.path.join(RESULTS_DIR, "table4_portfolio_es.csv"), index=False)
rolling.to_csv(os.path.join(PROC_DIR, "portfolio_es_rolling.csv"))
contingency[[
    "project_id", "ada", "type", "budget_eur", "duration_m",
    "es_crisis_type", "contingency_pct", "contingency_eur",
]].to_csv(os.path.join(PROC_DIR, "project_contingency.csv"),
          index=False, encoding="utf-8-sig")

for f in ("portfolio_es_by_regime.csv", "portfolio_es_rolling.csv",
          "project_contingency.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table4_portfolio_es.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 8. HEADLINE FINDING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Portfolio ES analysis complete")
print("=" * 64)

for w_label in ("equal", "budget"):
    sub = table4[table4["weights"] == w_label].set_index("regime")
    if not {"stable", "crisis"}.issubset(sub.index):
        continue
    db_s = sub.loc["stable", "div_benefit"]
    db_c = sub.loc["crisis", "div_benefit"]
    ratio_s = sub.loc["stable", "ratio"]
    ratio_c = sub.loc["crisis", "ratio"]
    print(f"\n  [{w_label} weights]")
    print(f"    Stable: ES_P/ES_Σ = {ratio_s:.3f}  (DB = {db_s:+.3f})")
    print(f"    Crisis: ES_P/ES_Σ = {ratio_c:.3f}  (DB = {db_c:+.3f})")
    if db_c < db_s:
        print(f"    → Diversification SHRINKS in crisis by "
              f"{(db_s - db_c)*100:.1f} pp")
    if db_c <= 0:
        print(f"    → CRISIS SUPER-ADDITIVITY:  Portfolio ES ≥ Σ individual ES")

print(f"\nNext: run 06_lstm_portfolio_agent.py")
print("=" * 64)
