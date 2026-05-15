"""
Paper 5 — Case studies + counterfactual €-gap + contingency rules
===================================
Three empirical-depth pieces that make the paper "speak" to the AiC
practitioner reviewer — ALL built from PUBLIC Διαύγεια data:

    1. THREE CRISIS CASE STUDIES
         (a) 2012 Greek sovereign crisis  (2011-07 → 2012-12)
         (b) 2015 capital controls        (2015-04 → 2015-12)
         (c) 2022 Ukraine energy shock    (2022-01 → 2023-06)
       For each: CSRI trajectory, which types became sources, ΔES vs
       surrounding stable periods.

    2. COUNTERFACTUAL €-GAP
       Translate the DB → 0 finding into a € figure over the total
       Διαύγεια portfolio budget.  A classical practitioner budgets
       contingency assuming DB_stable holds — we compute the crisis
       shortfall against the true ES_portfolio-based loss.

    3. CONTINGENCY RULE TABLE
       Distill the per-project `project_contingency.csv` into a compact
       rule table indexed by (type × duration bucket): a single
       contingency_pct that a PM can read off for a new project.

Inputs  (from 02–07):
    csri_monthly.csv                 — CSRI + regime labels
    contagion_flows.csv              — NET/OUT/IN flows per type
    contagion_index.csv              — CI_i(t)
    portfolio_es_by_regime.csv       — ES_sum, ES_P, DB by regime
    project_contingency.csv          — per-project contingency_pct
    portfolio_metadata.csv           — Διαύγεια budgets per project
    data/raw/diavgeia_projects.csv   — total portfolio budget

Outputs:
    data/processed/case_study_windows.csv
    data/processed/counterfactual_gap.csv
    data/processed/contingency_rules.csv
    results/table_case_studies.csv
    results/table_contingency_rules.csv
    results/figures/fig_case_studies.{pdf,png}
    results/figures/fig_counterfactual_gap.{pdf,png}
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROC_DIR    = "../data/processed/"
RAW_DIR     = "../data/raw/"
RESULTS_DIR = "../results/"
FIG_DIR     = RESULTS_DIR + "figures/"
os.makedirs(FIG_DIR, exist_ok=True)

TYPE_COLORS = {
    "road":     "#E74C3C",
    "bridge":   "#3498DB",
    "pipeline": "#27AE60",
    "building": "#F39C12",
    "other":    "#8E44AD",
}

print("=" * 64)
print("Paper 5 — Case studies + counterfactual + rules")
print("=" * 64)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading")
csri = pd.read_csv(PROC_DIR + "csri_monthly.csv",
                    index_col=0, parse_dates=True).sort_index()
flows = pd.read_csv(PROC_DIR + "contagion_flows.csv",
                     index_col=0, parse_dates=True).sort_index()
CI = pd.read_csv(PROC_DIR + "contagion_index.csv",
                  index_col=0, parse_dates=True).sort_index()
es_reg = pd.read_csv(PROC_DIR + "portfolio_es_by_regime.csv")
contingency = pd.read_csv(PROC_DIR + "project_contingency.csv")
meta = pd.read_csv(PROC_DIR + "portfolio_metadata.csv", encoding="utf-8-sig")
diavgeia = pd.read_csv(RAW_DIR + "diavgeia_projects.csv")

types = list(CI.columns)
print(f"  Types: {types}")
print(f"  CSRI coverage: {csri.index.min():%Y-%m} → {csri.index.max():%Y-%m}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CASE STUDIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Three crisis case studies")

CASES = [
    {"name":  "2012 Greek sovereign crisis",
     "code":  "GR2012",
     "start": "2011-07-01",
     "end":   "2012-12-01",
     "trigger": "PSI debt haircut (Mar 2012), second bailout",
     "color": "#C0392B"},
    {"name":  "2015 capital controls",
     "code":  "GR2015",
     "start": "2015-04-01",
     "end":   "2015-12-01",
     "trigger": "Referendum, 3-week bank holiday, €60/day limit",
     "color": "#2C3E50"},
    {"name":  "2022 Ukraine energy shock",
     "code":  "UKR2022",
     "start": "2022-01-01",
     "end":   "2023-06-01",
     "trigger": "Invasion (Feb 2022), gas/fuel/steel price spike",
     "color": "#1F618D"},
]

case_rows = []
for c in CASES:
    w_start = pd.Timestamp(c["start"])
    w_end   = pd.Timestamp(c["end"])
    window = csri.loc[(csri.index >= w_start) & (csri.index <= w_end)]
    if len(window) == 0:
        print(f"  ⚠ {c['code']}: window empty — skipped")
        continue

    # Baseline stable periods ±12M around the crisis window (before only)
    pre_start = w_start - pd.DateOffset(months=24)
    pre_end   = w_start - pd.DateOffset(months=1)
    pre_window = csri.loc[(csri.index >= pre_start) & (csri.index <= pre_end)]

    # Which types were sources during the window?
    net_cols = [f"NET_{t}" for t in types if f"NET_{t}" in flows.columns]
    flow_window = flows[net_cols].reindex(window.index).dropna()
    if len(flow_window):
        mean_net = flow_window.mean()
        # Rank by NET — sources = positive, receivers = negative
        sources = mean_net.sort_values(ascending=False)
        top_source = sources.index[0].replace("NET_", "")
        top_receiver = sources.index[-1].replace("NET_", "")
    else:
        top_source = top_receiver = "n/a"

    case_rows.append({
        "code":            c["code"],
        "name":            c["name"],
        "start":           w_start.strftime("%Y-%m"),
        "end":             w_end.strftime("%Y-%m"),
        "trigger":         c["trigger"],
        "n_months":        len(window),
        "CSRI_pre":        float(pre_window["CSRI"].mean()) if len(pre_window) else np.nan,
        "CSRI_crisis":     float(window["CSRI"].mean()),
        "CSRI_peak":       float(window["CSRI"].max()),
        "CSRI_peak_date":  window["CSRI"].idxmax().strftime("%Y-%m") if window["CSRI"].notna().any() else "",
        "lambdaU_crisis":  float(window["lambdaU"].mean()),
        "lambdaU_pre":     float(pre_window["lambdaU"].mean()) if len(pre_window) else np.nan,
        "top_source":      top_source,
        "top_receiver":    top_receiver,
    })

case_df = pd.DataFrame(case_rows)
print(case_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. COUNTERFACTUAL €-GAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Counterfactual €-gap (naive vs contagion ES)")

# Use budget-weighted row from portfolio_es_by_regime
es_budget = es_reg[es_reg["weights"] == "budget"].set_index("regime")

total_budget = float(diavgeia["budget_eur"].sum())
print(f"  Total Διαύγεια portfolio budget: €{total_budget:,.0f}")
print("\n  The €-gap story:")
print("    A practitioner using classical diversification assumes DB_stable")
print("    will HOLD under any regime — i.e. they budget contingency for")
print("    (1 − DB_stable) × ES_sum.  Under crisis DB → 0 and the portfolio")
print("    loss is ES_sum × total_budget (no saving), so the 'lost")
print("    diversification benefit' is the money the practitioner THOUGHT")
print("    they would save and which evaporates exactly when they need it.\n")

# Lookup DB in stable (what the practitioner naively assumes)
db_stable = float(es_budget.loc["stable", "div_benefit"])
es_sum_stable = float(es_budget.loc["stable", "ES_sum"])
es_sum_crisis = float(es_budget.loc["crisis", "ES_sum"])
es_p_crisis   = float(es_budget.loc["crisis", "ES_portfolio"])

gap_rows = []
for regime in ("full", "stable", "crisis"):
    if regime not in es_budget.index:
        continue
    row = es_budget.loc[regime]
    es_sum = float(row["ES_sum"])
    es_p   = float(row["ES_portfolio"])

    # (A) Classical practitioner path: believes DB = DB_stable always, so
    #     their BUDGETED contingency is (1 − DB_stable) × ES_sum × budget
    naive_budgeted_eur = (1.0 - db_stable) * es_sum * total_budget

    # (B) Reality: true loss = ES_portfolio × budget
    true_loss_eur = es_p * total_budget

    # (C) Shortfall when reality > naive budget (positive = under-budgeted)
    shortfall_eur = true_loss_eur - naive_budgeted_eur
    shortfall_pct = (shortfall_eur / naive_budgeted_eur * 100.0
                      if naive_budgeted_eur > 0 else np.nan)

    gap_rows.append({
        "regime":             regime,
        "ES_sum":             es_sum,
        "ES_P":               es_p,
        "DB_actual":          float(row["div_benefit"]),
        "DB_assumed":         db_stable,
        "naive_budget_eur":   naive_budgeted_eur,
        "true_loss_eur":      true_loss_eur,
        "shortfall_eur":      shortfall_eur,
        "shortfall_pct":      shortfall_pct,
    })

gap_df = pd.DataFrame(gap_rows)
print(gap_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# Headline €-gap number for the paper
crisis_sf = gap_df.loc[gap_df["regime"] == "crisis", "shortfall_eur"]
if len(crisis_sf):
    headline_eur = float(crisis_sf.iloc[0])
    print(f"\n  🟥 HEADLINE: Crisis budget shortfall vs classical assumption = "
          f"€{headline_eur:,.0f}  ({headline_eur / 1e6:+.2f} M€)")
    print(f"      That is the money the classical PM would be MISSING from")
    print(f"      their contingency reserve at peak crisis.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTINGENCY RULE TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Contingency rule table (type × duration bucket)")

# Duration buckets: <18M, 18-30M, 30-48M, >48M
def bucket(d):
    if d < 18:   return "short (<18M)"
    if d < 30:   return "medium (18-30M)"
    if d < 48:   return "long (30-48M)"
    return "xlong (>48M)"

contingency["duration_bucket"] = contingency["duration_m"].map(bucket)

# Aggregate: median contingency_pct per (type, bucket)
rule = (contingency
         .groupby(["type", "duration_bucket"])
         .agg(n=("contingency_pct", "count"),
              median_pct=("contingency_pct", "median"),
              mean_pct=("contingency_pct", "mean"),
              p75_pct=("contingency_pct", lambda s: s.quantile(0.75)),
              mean_budget=("budget_eur", "mean"))
         .reset_index())
rule["rule_pct"] = (rule["p75_pct"] * 100).round(1)      # use P75 as "safe" rule
rule["rule_text"] = rule.apply(
    lambda r: f"If type={r['type']:8s} AND duration={r['duration_bucket']:15s} "
               f"→ add {r['rule_pct']:4.1f}% contingency  (n={int(r['n'])})",
    axis=1)

print(rule[["type", "duration_bucket", "n",
             "median_pct", "p75_pct", "rule_pct"]]
       .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n  Rule table (first 8 rules):")
for t in rule["rule_text"].head(8):
    print(f"    {t}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "savefig.bbox": "tight"})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR + f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}{name}.{{pdf,png}}")


# ------------------------------------------------------------------ CASE STUDIES
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
for ax, c in zip(axes, CASES):
    w_start = pd.Timestamp(c["start"])
    w_end   = pd.Timestamp(c["end"])
    pad_start = w_start - pd.DateOffset(months=12)
    pad_end   = w_end   + pd.DateOffset(months=12)
    sub = csri.loc[(csri.index >= pad_start) & (csri.index <= pad_end)]
    if len(sub):
        ax.plot(sub.index, sub["CSRI"], lw=1.4, color="#2C3E50", label="CSRI")
        ax.axvspan(w_start, w_end, color=c["color"], alpha=0.18,
                   label="crisis window")
    ax.axhline(0, color="#7F8C8D", lw=0.6, ls="--")
    ax.set_title(c["name"], fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=7)
axes[0].set_ylabel("CSRI (σ units)")
fig.suptitle("Figure — Three crisis case studies (CSRI trajectory)",
              fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "fig_case_studies")


# ------------------------------------------------------------ COUNTERFACTUAL GAP
if not gap_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(gap_df))
    w = 0.38
    ax.bar(x - w / 2, gap_df["naive_budget_eur"] / 1e6, w,
           label="Naive budget  ((1−DB_stable)·ES_Σ·B)", color="#3498DB")
    ax.bar(x + w / 2, gap_df["true_loss_eur"] / 1e6, w,
           label="True loss  (ES_P·B)", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_df["regime"])
    ax.set_ylabel("Expected € (€M)")
    ax.set_title("Figure — Counterfactual €-gap:  contingency budget vs true loss",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    for i, row in gap_df.iterrows():
        sf = row["shortfall_eur"] / 1e6
        top = max(row["naive_budget_eur"], row["true_loss_eur"]) / 1e6
        ax.annotate(f"shortfall = €{sf:+,.2f} M",
                    xy=(i, top), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, fontweight="bold",
                    color="#C0392B" if sf > 0 else "#27AE60")
    fig.tight_layout()
    _save(fig, "fig_counterfactual_gap")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving CSVs")

case_df.to_csv(PROC_DIR + "case_study_windows.csv", index=False)
case_df.to_csv(RESULTS_DIR + "table_case_studies.csv", index=False)

gap_df.to_csv(PROC_DIR + "counterfactual_gap.csv", index=False)

rule.to_csv(PROC_DIR + "contingency_rules.csv", index=False)
rule[["type", "duration_bucket", "n", "rule_pct", "rule_text"]].to_csv(
    RESULTS_DIR + "table_contingency_rules.csv", index=False)

for f in ("case_study_windows.csv", "counterfactual_gap.csv",
           "contingency_rules.csv"):
    print(f"  Saved: {PROC_DIR}{f}")
print(f"  Saved: {RESULTS_DIR}table_case_studies.csv")
print(f"  Saved: {RESULTS_DIR}table_contingency_rules.csv")


print("\n" + "=" * 64)
print("DONE — Group 4 complete")
print("=" * 64)
if not gap_df.empty and "crisis" in set(gap_df["regime"]):
    g = gap_df.loc[gap_df["regime"] == "crisis"].iloc[0]
    print(f"  Crisis shortfall vs naive budget: €{g['shortfall_eur']/1e6:+.2f} M "
          f"  ({g['shortfall_pct']:+.2f}%)")
print(f"  Case studies: {len(case_df)} crisis episodes analysed")
print(f"  Contingency rules: {len(rule)} (type × duration) buckets")
print("=" * 64)
