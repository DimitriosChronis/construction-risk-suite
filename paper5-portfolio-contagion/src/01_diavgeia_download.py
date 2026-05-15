"""
Paper 5 — Διαύγεια Data Download
===================================
Downloads Greek public construction contract awards from the
Διαύγεια open data API (diavgeia.gov.gr).

Target: 500+ construction projects with:
  - Budget >= EUR 100,000
  - Period: 2011–2024
  - Decision types:
      Δ.1  (ΑΝΑΘΕΣΗ ΕΡΓΩΝ)  → filtered by assignmentType="Έργα"
      Δ.2.2 (ΚΑΤΑΚΥΡΩΣΗ)    → filtered by subject keywords (construction)

Outputs:
  - data/raw/diavgeia_projects.csv     (cleaned project table)
  - data/raw/diavgeia_summary.csv      (summary stats)
  - data/raw/diavgeia_raw_awards.jsonl (raw records for audit)

Usage:
  python 01_diavgeia_download.py               # full run 2011-2024
  python 01_diavgeia_download.py --test        # quick test: 2020 Q1 only
  python 01_diavgeia_download.py --since 2018  # from given year

API notes (verified 2026-04-14):
  - Endpoint: https://diavgeia.gov.gr/luminapi/opendata/search.json
  - Filter via `type` URL param.
  - Date window via `from_issue_date` / `to_issue_date` — walk MONTHLY.
  - Pagination via `page` (0-indexed) and `size` (max 500).
  - Δ.1 covers works/supplies/services/studies — ~5% are "Έργα",
    but mostly small budgets (direct awards < €60K).
  - Δ.2.2 (ΚΑΤΑΚΥΡΩΣΗ) covers competitive-tender awards — contains
    the large construction projects but has no `assignmentType` field,
    so we filter by Greek construction keywords in `subject`.
  - `extraFieldValues.awardAmount` is nested {amount, currency}.
"""

import argparse
import io
import json
import os
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

# Force UTF-8 console output on Windows (Greek labels)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
RAW_DIR      = "../data/raw/"
RESULTS_DIR  = "../results/"
BASE_URL     = "https://diavgeia.gov.gr/luminapi/opendata/search.json"

# Decision types to scan.
# Δ.2.2 (ΚΑΤΑΚΥΡΩΣΗ) = competitive tender awards — has the large construction
# projects (~3K records/month, fast).
# Δ.1 (ΑΝΑΘΕΣΗ) = direct awards — mostly small budgets, 20-40K records/month
# (very slow to paginate). Enable only if you need small projects too.
DECISION_TYPES = ["Δ.2.2"]     # add "Δ.1" for direct awards (slow, mostly <€100K)

MIN_BUDGET   = 100_000         # EUR (lowered from 500K — filter later)
START_YEAR   = 2014            # Διαύγεια sparse before 2014; Δ.2.2 reliable from ~2013
END_YEAR     = 2024
PAGE_SIZE    = 500             # max per request
MAX_PAGES_PER_WINDOW = 40      # ES deep-pagination cap (~20 000 offset)
SLEEP_SEC    = 0.3             # rate limiting
REQUEST_TIMEOUT = 60

# Construction keywords for subject-based filtering (Δ.2.2)
CONSTRUCTION_KEYWORDS = [
    "οδό", "οδοποι", "δρόμο", "ασφαλτ", "ασφαλτό",
    "γέφυρ",
    "δίκτυ", "αγωγ", "ύδρευσ", "αποχέτευσ",
    "σχολ", "νοσοκομ", "κτίρι", "ανέγερσ",
    "κατασκευ", "έργου", "έργο ",
    "τεχνικ", "αποκατάστασ", "συντήρησ", "ανακαίνισ",
    "επισκευ", "ανάπλασ", "αναπλασ", "διαμόρφ",
    "πεζοδρόμ", "αντιπλημμ", "αποστράγγισ",
]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true",
                    help="Quick test: 2020 Q1 only")
parser.add_argument("--since", type=int, default=START_YEAR,
                    help=f"Start year (default {START_YEAR})")
parser.add_argument("--until", type=int, default=END_YEAR,
                    help=f"End year (default {END_YEAR})")
args = parser.parse_args()

def month_window(year: int, month: int) -> tuple[date, date]:
    last = monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


if args.test:
    windows = [month_window(2020, m) for m in range(1, 4)]
    print("** TEST MODE: 2020-01..2020-03 only **")
else:
    windows = [month_window(y, m)
               for y in range(args.since, args.until + 1)
               for m in range(1, 13)]

print("=" * 64)
print("Paper 5 — Διαύγεια Construction Awards Download")
print(f"Period: {args.since}–{args.until}  |  min budget: EUR {MIN_BUDGET:,}")
print(f"Decision types: {', '.join(DECISION_TYPES)}")
for dt in DECISION_TYPES:
    if dt == "Δ.1":
        print(f"  Δ.1   → filter: assignmentType='Έργα'")
    elif dt == "Δ.2.2":
        print(f"  Δ.2.2 → filter: subject keywords (construction)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def extract_budget(extra: dict) -> float:
    aw = extra.get("awardAmount")
    if isinstance(aw, dict):
        amt = aw.get("amount")
        try:
            return float(amt) if amt is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def is_construction_decision(d: dict, decision_type: str) -> bool:
    """Check whether a Διαύγεια decision is a construction project."""
    extra = d.get("extraFieldValues", {}) or {}

    if decision_type == "Δ.1":
        # Δ.1 has explicit assignmentType field
        return extra.get("assignmentType") == "Έργα"

    # Δ.2.2 has no assignmentType — match on subject keywords
    subj = (d.get("subject") or "").lower()
    return any(kw in subj for kw in CONSTRUCTION_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD — walk months × decision types, paginate, filter
# ══════════════════════════════════════════════════════════════════════════════
def fetch_window(d_from: date, d_to: date, decision_type: str,
                 session: requests.Session) -> tuple[list[dict], int, bool]:
    """Fetch construction records in [d_from, d_to] meeting MIN_BUDGET.

    Returns (hits, server_total, truncated).
    """
    hits = []
    page = 0
    total = None
    while page < MAX_PAGES_PER_WINDOW:
        params = {
            "q":               "*",
            "type":            decision_type,
            "from_issue_date": d_from.isoformat(),
            "to_issue_date":   d_to.isoformat(),
            "size":            PAGE_SIZE,
            "page":            page,
        }
        j = None
        for attempt in range(4):
            try:
                r = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                j = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                wait = 2 ** attempt * 5
                print(f"    ! {d_from}→{d_to} page {page} try {attempt+1}/4: {e}"
                      f"  (retry in {wait}s)", flush=True)
                time.sleep(wait)
        if j is None:
            print(f"    ✗ {d_from}→{d_to} page {page} failed after 4 retries; skipping",
                  flush=True)
            page += 1
            continue

        if total is None:
            total = j.get("info", {}).get("total", 0)

        decisions = j.get("decisions", [])
        if not decisions:
            break

        for d in decisions:
            if not is_construction_decision(d, decision_type):
                continue
            extra = d.get("extraFieldValues", {}) or {}
            budget = extract_budget(extra)
            if budget < MIN_BUDGET:
                continue
            hits.append({
                "ada":           d.get("ada", ""),
                "issue_date":    d.get("issueDate"),
                "submit_date":   d.get("submissionTimestamp"),
                "decision_type": decision_type,
                "type":          d.get("decisionTypeId", ""),
                "subject":       (d.get("subject") or "")[:500],
                "organization":  d.get("organizationId", ""),
                "unit":          ",".join(d.get("unitIds", []) or []),
                "budget_eur":    budget,
                "currency":      (extra.get("awardAmount") or {}).get("currency", "EUR"),
                "cpv":           ",".join(extra.get("cpv") or []),
                "status":        d.get("status", ""),
                "url":           d.get("documentUrl", ""),
                "_raw":          d,          # kept only for JSONL audit dump
            })

        page += 1
        # Stop if we've exhausted the server-side result set
        if (page * PAGE_SIZE) >= (total or 0):
            break
        time.sleep(SLEEP_SEC)

    truncated = (total or 0) > page * PAGE_SIZE
    return hits, (total or 0), truncated


def fetch_window_recursive(d_from: date, d_to: date, decision_type: str,
                            session: requests.Session,
                            depth: int = 0) -> list[dict]:
    """Fetch a window, auto-splitting in half if the server truncates."""
    hits, total, truncated = fetch_window(d_from, d_to, decision_type, session)
    label = f"{d_from}→{d_to}"
    indent = "  " * depth
    if truncated and (d_to - d_from).days >= 1:
        print(f"    {indent}{label}: total={total:>6}  [split]", flush=True)
        mid = d_from + (d_to - d_from) / 2
        left  = fetch_window_recursive(d_from, mid, decision_type, session, depth + 1)
        right = fetch_window_recursive(mid + timedelta(days=1), d_to, decision_type, session, depth + 1)
        return left + right
    if hits:
        print(f"    {indent}{label}: total={total:>6}  kept={len(hits):>4}", flush=True)
    return hits


print("\nSTEP 1: Downloading awards (monthly, auto-split on truncation)")
print("-" * 64)

session = requests.Session()
session.headers.update({"Accept": "application/json",
                        "User-Agent": "construction-risk-paper5/1.0"})

all_records: list[dict] = []
seen_ada: set[str] = set()

total_windows = len(windows)
t0 = time.time()

for i, (d_from, d_to) in enumerate(windows):
    pct = (i / total_windows) * 100
    elapsed = time.time() - t0
    eta_min = (elapsed / max(i, 1)) * (total_windows - i) / 60 if i > 0 else 0
    print(f"\n[{i+1}/{total_windows}] {d_from.strftime('%Y-%m')} "
          f"| {pct:.0f}% "
          f"| projects: {len(all_records)} "
          f"| elapsed: {elapsed/60:.1f}m "
          f"| ETA: {eta_min:.0f}m",
          flush=True)

    for dtype in DECISION_TYPES:
        batch = fetch_window_recursive(d_from, d_to, dtype, session)
        for rec in batch:
            ada = rec["ada"]
            if ada and ada not in seen_ada:
                seen_ada.add(ada)
                all_records.append(rec)

print(f"\n  Unique construction awards collected: {len(all_records)}")

if not all_records:
    print("\n  ✗ No records downloaded. Aborting — check API connectivity.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DUMP RAW JSONL (audit trail) + BUILD DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Writing raw audit dump")
raw_path = os.path.join(RAW_DIR, "diavgeia_raw_awards.jsonl")
with open(raw_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec["_raw"], ensure_ascii=False) + "\n")
print(f"  Saved: {raw_path}")

# Strip raw from the frame
for rec in all_records:
    rec.pop("_raw", None)

df = pd.DataFrame(all_records)

# Convert epoch-ms timestamps to datetimes
for col in ("issue_date", "submit_date"):
    df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
df["date"] = df["issue_date"]
df["year"] = df["date"].dt.year

df["budget_eur"] = pd.to_numeric(df["budget_eur"], errors="coerce").fillna(0)
df = df[df["budget_eur"] >= MIN_BUDGET].copy()
df = df.dropna(subset=["date"]).copy()
df = df.sort_values("date").reset_index(drop=True)

df["budget_log"] = np.log(df["budget_eur"])
df["budget_category"] = pd.cut(
    df["budget_eur"],
    bins=[0, 5e5, 1e6, 5e6, 20e6, np.inf],
    labels=["Small (100-500K)", "Medium (0.5-1M)",
            "Large (1-5M)", "Major (5-20M)", "Mega (>20M)"],
)

print(f"\n  Projects after cleaning: {len(df)}")
print(f"  Period: {df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}")
print(f"  Budget range: EUR {df['budget_eur'].min():,.0f} → "
      f"EUR {df['budget_eur'].max():,.0f}")
print(f"  Median budget: EUR {df['budget_eur'].median():,.0f}")
print(f"\n  By decision type:")
print(df["decision_type"].value_counts().to_string())
print(f"\n  Budget categories:")
print(df["budget_category"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3. ASSIGN MATERIAL EXPOSURE WEIGHTS (from project subject)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Assigning material exposure weights")


def assign_material_weights(subject: str) -> dict[str, float]:
    """Assign material weights based on project description (sums to 1.0)."""
    s = str(subject).lower()
    if any(w in s for w in ["οδό", "οδοποι", "δρόμο", "ασφαλτ", "road", "highway"]):
        return {"concrete": 0.35, "steel": 0.20, "fuel": 0.35, "pvc": 0.10}
    if any(w in s for w in ["γέφυρ", "bridge"]):
        return {"concrete": 0.30, "steel": 0.45, "fuel": 0.15, "pvc": 0.10}
    if any(w in s for w in ["δίκτυ", "αγωγ", "ύδρευσ", "αποχέτευσ", "pipeline", "network"]):
        return {"concrete": 0.20, "steel": 0.20, "fuel": 0.25, "pvc": 0.35}
    if any(w in s for w in ["σχολ", "νοσοκομ", "school", "hospital", "κτίρι", "ανέγερσ"]):
        return {"concrete": 0.40, "steel": 0.30, "fuel": 0.15, "pvc": 0.15}
    return {"concrete": 0.30, "steel": 0.30, "fuel": 0.20, "pvc": 0.20}


weights = df["subject"].map(assign_material_weights)
for mat in ("concrete", "steel", "fuel", "pvc"):
    df[f"w_{mat}"] = [w[mat] for w in weights]

print("  Mean weights:")
for mat in ("concrete", "steel", "fuel", "pvc"):
    print(f"    {mat:10s}: {df[f'w_{mat}'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Saving")

out_csv = os.path.join(RAW_DIR, "diavgeia_projects.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"  Saved: {out_csv}  ({len(df)} rows)")

summary = {
    "download_date":   datetime.now().strftime("%Y-%m-%d"),
    "api_endpoint":    BASE_URL,
    "decision_types":  ", ".join(DECISION_TYPES),
    "total_projects":  len(df),
    "period_start":    df["date"].min().strftime("%Y-%m"),
    "period_end":      df["date"].max().strftime("%Y-%m"),
    "min_budget_eur":  df["budget_eur"].min(),
    "max_budget_eur":  df["budget_eur"].max(),
    "median_budget":   df["budget_eur"].median(),
    "total_value_eur": df["budget_eur"].sum(),
}
pd.DataFrame([summary]).to_csv(
    os.path.join(RAW_DIR, "diavgeia_summary.csv"), index=False,
    encoding="utf-8-sig",
)
print(f"  Saved: {RAW_DIR}diavgeia_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Διαύγεια download complete")
print("=" * 64)
print(f"  Total projects: {len(df)}")
print(f"  Total value:    EUR {df['budget_eur'].sum():,.0f}")
print(f"  Period:         {summary['period_start']} → {summary['period_end']}")
print(f"\nNext: run 02_portfolio_construction.py")
print("=" * 64)
