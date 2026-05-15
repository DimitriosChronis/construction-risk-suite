"""
Paper 5 — Portfolio Construction
===================================
Builds the N×T project portfolio matrix that downstream scripts
(03 dynamic copula, 04 contagion, 05 portfolio ES) consume.

Inputs:
  - data/raw/diavgeia_projects.csv                      (from 01)
  - ../../paper3-es-hedging/data/processed/elstat_log_returns.csv
        (shared ELSTAT monthly log-returns: concrete/steel/fuel/pvc)

Outputs:
  - data/processed/portfolio_project_returns.csv   (months × project_id, long)
  - data/processed/portfolio_project_returns_wide.csv  (months × project_id)
  - data/processed/portfolio_type_returns.csv     (months × project type, 5 cols)
  - data/processed/portfolio_metadata.csv         (project metadata + weights)
  - data/processed/portfolio_summary.csv          (Table 1 of paper)

Method:
  1. Classify each project into a TYPE from the subject text
       {road, building, pipeline, bridge, other}
  2. Assign fixed material weights per type (from Paper 3 baseline).
  3. Estimate a duration in months per project from its budget
       duration = clip(12 + 6·log10(budget / MIN_BUDGET), 12, 48)
  4. Project 'return' at month t is the negative weighted log-change in
     material prices (cost increase → negative portfolio return):
         r_i(t) = -Σ_k w_ik · Δlog P_k(t)
     It is only defined inside the project's active window
     [issue_date, issue_date + duration].
  5. For the dynamic copula, we aggregate to TYPE level: the equally-weighted
     average of all active projects of a given type at month t.

Run from: construction-risk-suite/paper5-portfolio-contagion/src/
"""

import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
RAW_DIR       = "../data/raw/"
PROC_DIR      = "../data/processed/"
ELSTAT_PATH   = "../../paper3-es-hedging/data/processed/elstat_log_returns.csv"
MIN_BUDGET    = 500_000
MAX_BUDGET    = 500_000_000  # EUR 500M cap — anything above is data-entry error
DUR_MIN, DUR_MAX = 12, 48   # months
os.makedirs(PROC_DIR, exist_ok=True)

# Material weights per project type (must sum to 1.0)
TYPE_WEIGHTS = {
    "road":     {"concrete": 0.35, "steel": 0.20, "fuel": 0.35, "pvc": 0.10},
    "bridge":   {"concrete": 0.30, "steel": 0.45, "fuel": 0.15, "pvc": 0.10},
    "pipeline": {"concrete": 0.20, "steel": 0.20, "fuel": 0.25, "pvc": 0.35},
    "building": {"concrete": 0.40, "steel": 0.30, "fuel": 0.15, "pvc": 0.15},
    "other":    {"concrete": 0.30, "steel": 0.30, "fuel": 0.20, "pvc": 0.20},
}
TYPES = list(TYPE_WEIGHTS.keys())
MATERIALS = ["concrete", "steel", "fuel", "pvc"]

# ELSTAT column names (from P3 shared processing)
ELSTAT_COLS = {
    "concrete": "GR_Concrete",
    "steel":    "GR_Steel",
    "fuel":     "GR_Fuel_Energy",
    "pvc":      "GR_PVC_Pipes",
}

print("=" * 64)
print("Paper 5 — Portfolio Construction")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD INPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading inputs")

projects_path = os.path.join(RAW_DIR, "diavgeia_projects.csv")
if not os.path.exists(projects_path):
    sys.exit(f"  ✗ Missing {projects_path} — run 01_diavgeia_download.py first")

proj = pd.read_csv(projects_path, encoding="utf-8-sig")
proj["date"] = pd.to_datetime(proj["date"], errors="coerce")
proj = proj.dropna(subset=["date", "budget_eur"]).reset_index(drop=True)

# Budget filters
n_raw = len(proj)
proj = proj[(proj["budget_eur"] >= MIN_BUDGET) &
            (proj["budget_eur"] <= MAX_BUDGET)].reset_index(drop=True)
n_dropped = n_raw - len(proj)
if n_dropped:
    print(f"  Dropped {n_dropped} projects outside EUR {MIN_BUDGET:,.0f}–{MAX_BUDGET:,.0f}")

print(f"  Projects: {len(proj)}  ({proj['date'].min():%Y-%m} → "
      f"{proj['date'].max():%Y-%m})")
print(f"  Budget: EUR {proj['budget_eur'].min():,.0f} → EUR {proj['budget_eur'].max():,.0f}"
      f"  (median: EUR {proj['budget_eur'].median():,.0f})")

if not os.path.exists(ELSTAT_PATH):
    sys.exit(f"  ✗ Missing ELSTAT file: {ELSTAT_PATH}")
elstat = pd.read_csv(ELSTAT_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
elstat = elstat[list(ELSTAT_COLS.values())].rename(
    columns={v: k for k, v in ELSTAT_COLS.items()}
)
# Normalize to month-start
elstat.index = elstat.index.to_period("M").to_timestamp()
print(f"  ELSTAT:   {len(elstat)} months  "
      f"({elstat.index.min():%Y-%m} → {elstat.index.max():%Y-%m})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFY PROJECT TYPE + RE-ASSIGN WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Classifying project types")


def classify(subject: str) -> str:
    s = str(subject).lower()
    if any(w in s for w in ["οδό", "οδοποι", "δρόμο", "ασφαλτ", "road", "highway"]):
        return "road"
    if any(w in s for w in ["γέφυρ", "bridge"]):
        return "bridge"
    if any(w in s for w in ["δίκτυ", "αγωγ", "ύδρευσ", "αποχέτευσ", "pipeline", "network"]):
        return "pipeline"
    if any(w in s for w in ["σχολ", "νοσοκομ", "school", "hospital",
                             "κτίρι", "ανέγερσ", "οικοδ"]):
        return "building"
    return "other"


proj["type"] = proj["subject"].map(classify)
for mat in MATERIALS:
    proj[f"w_{mat}"] = proj["type"].map(lambda t: TYPE_WEIGHTS[t][mat])

print(proj["type"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3. DURATION ESTIMATE  + ACTIVE WINDOW
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Estimating project durations")


def estimate_duration_months(budget_eur: float) -> int:
    """Log-linear scale: 500 k → 12 m, 5 M → 18 m, 50 M → 24 m, capped 48."""
    m = 12 + 6 * np.log10(max(budget_eur, MIN_BUDGET) / MIN_BUDGET)
    return int(np.clip(round(m), DUR_MIN, DUR_MAX))


proj["duration_m"] = proj["budget_eur"].map(estimate_duration_months)
proj["start_month"] = proj["date"].dt.to_period("M").dt.to_timestamp()
proj["end_month"] = proj.apply(
    lambda r: (r["start_month"]
               + pd.offsets.MonthBegin(r["duration_m"] - 1)),
    axis=1,
)
print(f"  Duration: mean={proj['duration_m'].mean():.1f}m, "
      f"median={proj['duration_m'].median():.0f}m, "
      f"min={proj['duration_m'].min()}, max={proj['duration_m'].max()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD PROJECT-LEVEL RETURNS (wide + long)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Building N×T project returns matrix")

# Assign a short project_id
proj = proj.sort_values(["start_month", "ada"]).reset_index(drop=True)
proj["project_id"] = [f"P{i:05d}" for i in range(len(proj))]

# Restrict ELSTAT to the span that covers any project
span_start = proj["start_month"].min()
span_end   = proj["end_month"].max()
elstat_span = elstat.loc[
    (elstat.index >= span_start) & (elstat.index <= span_end)
].copy()

if elstat_span.empty:
    sys.exit("  ✗ No ELSTAT data overlaps project window — check date alignment")

print(f"  Span: {span_start:%Y-%m} → {span_end:%Y-%m}  "
      f"({len(elstat_span)} months)")

# For every project compute a monthly "cost shock" series (= -weighted material return)
# Vectorised: W (N×K) @ R.T (K×T) → N×T
W = proj[[f"w_{m}" for m in MATERIALS]].to_numpy()               # N × 4
R = elstat_span[MATERIALS].to_numpy()                            # T × 4
shock_full = -(W @ R.T)                                          # N × T  (neg = cost up)

wide = pd.DataFrame(
    shock_full.T,                                                # T × N
    index=elstat_span.index,
    columns=proj["project_id"].to_list(),
)
wide.index.name = "date"

# Mask out inactive months per project
for pid, s, e in zip(proj["project_id"], proj["start_month"], proj["end_month"]):
    mask = (wide.index < s) | (wide.index > e)
    wide.loc[mask, pid] = np.nan

# Drop all-NaN columns (shouldn't happen if span is correct)
wide = wide.dropna(axis=1, how="all")

active_counts = wide.notna().sum(axis=1)
print(f"  Active projects per month — mean: {active_counts.mean():.1f}, "
      f"peak: {active_counts.max()}")

# Long format for convenience
long = (
    wide.reset_index()
        .melt(id_vars="date", var_name="project_id", value_name="ret")
        .dropna(subset=["ret"])
)
print(f"  Long table: {len(long):,} (project × month) observations")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD TYPE-LEVEL AGGREGATE RETURNS  (for dynamic copula in 03)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Aggregating to project-type level")

# At month t, type_return(type) = mean of shocks across active projects of that type
type_returns = pd.DataFrame(index=elstat_span.index,
                             columns=TYPES, dtype=float)

type_of = proj.set_index("project_id")["type"].to_dict()
for t in TYPES:
    cols = [pid for pid in wide.columns if type_of.get(pid) == t]
    if cols:
        type_returns[t] = wide[cols].mean(axis=1)
    else:
        type_returns[t] = np.nan

# For 03 copula we need a theoretical "always-on" type series covering
# the FULL ELSTAT history (not just the project span), so that the rolling
# 24-month copula has enough observations regardless of how many projects
# were actually downloaded.
type_theoretical = pd.DataFrame(index=elstat.index,
                                 columns=TYPES, dtype=float)
R_full = elstat[MATERIALS].to_numpy()
for t in TYPES:
    w = np.array([TYPE_WEIGHTS[t][m] for m in MATERIALS])
    type_theoretical[t] = -(R_full @ w)

print(f"  Empirical type returns (active only) — coverage:")
for t in TYPES:
    cov = type_returns[t].notna().sum()
    print(f"    {t:9s}: {cov} months")

print(f"  Theoretical type returns — coverage: {len(type_theoretical)} months "
      f"({type_theoretical.index.min():%Y-%m} → "
      f"{type_theoretical.index.max():%Y-%m})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. METADATA + SUMMARY (Table 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Summary statistics")

meta = proj[[
    "project_id", "ada", "type", "subject", "organization",
    "budget_eur", "budget_category", "duration_m",
    "start_month", "end_month",
    "w_concrete", "w_steel", "w_fuel", "w_pvc",
]].copy()

table1 = {
    "n_projects":           len(proj),
    "period_start":         f"{span_start:%Y-%m}",
    "period_end":           f"{span_end:%Y-%m}",
    "total_months":         len(elstat_span),
    "total_budget_eur":     proj["budget_eur"].sum(),
    "median_budget_eur":    proj["budget_eur"].median(),
    "mean_duration_months": proj["duration_m"].mean(),
    "mean_active_per_month": active_counts.mean(),
    "peak_active_per_month": int(active_counts.max()),
}
for t in TYPES:
    table1[f"n_{t}"] = int((proj["type"] == t).sum())
for mat in MATERIALS:
    table1[f"mean_w_{mat}"] = proj[f"w_{mat}"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Saving outputs")

wide.to_csv(os.path.join(PROC_DIR, "portfolio_project_returns_wide.csv"))
long.to_csv(os.path.join(PROC_DIR, "portfolio_project_returns.csv"), index=False)
type_returns.to_csv(os.path.join(PROC_DIR, "portfolio_type_returns.csv"))
type_theoretical.to_csv(os.path.join(PROC_DIR, "portfolio_type_returns_theoretical.csv"))
meta.to_csv(os.path.join(PROC_DIR, "portfolio_metadata.csv"),
            index=False, encoding="utf-8-sig")
pd.DataFrame([table1]).to_csv(
    os.path.join(PROC_DIR, "portfolio_summary.csv"), index=False
)

for f in ("portfolio_project_returns_wide.csv",
          "portfolio_project_returns.csv",
          "portfolio_type_returns.csv",
          "portfolio_type_returns_theoretical.csv",
          "portfolio_metadata.csv",
          "portfolio_summary.csv"):
    print(f"  Saved: {PROC_DIR}{f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Portfolio construction complete")
print("=" * 64)
for k, v in table1.items():
    if isinstance(v, float):
        print(f"  {k:<25s}: {v:,.3f}")
    else:
        print(f"  {k:<25s}: {v}")
print(f"\nNext: run 03_dynamic_copula.py")
print("=" * 64)
