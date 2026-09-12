"""
01_eurostat_cci.py
==================
Paper 6 primary data downloader (v3).

Downloads monthly construction-cost indices (Eurostat sts_copi_m,
indicator COST, unit I21, NSA, residential buildings) for the seven
non-Greek panel countries, the PRC_PRR robustness set, and the
quarterly layer (sts_copi_q) for DE/FR/EL. Greece (monthly) comes
from the in-house ELSTAT SPC23 pipeline (Paper 2 aligned dataset),
cached locally and refreshed manually when ELSTAT publishes.

Outputs:
    data/raw/eurostat_cci_monthly_COST.csv       (long: date,country,level)
    data/raw/eurostat_cci_monthly_PRC_PRR.csv
    data/raw/eurostat_cci_quarterly.csv
    data/raw/cci_GR_elstat.csv                   (date,level)

Primary panel countries (monthly, COST):
    Mediterranean: GR(ELSTAT), ES, IT, PT
    Northern:      NL, IE, LT, NO
"""

import os
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

MONTHLY_COST_GEO = ["ES", "IT", "PT", "NL", "IE", "LT", "NO"]
MONTHLY_PRR_GEO = ["ES", "IT", "PT", "NL", "IE", "LT", "NO", "FI", "PL"]
QUARTERLY_GEO = ["DE", "FR", "EL"]

COMMON = {"format": "JSON", "lang": "EN", "unit": "I21", "s_adj": "NSA"}

# Preferred CPA categories, most-specific first; per country we keep the
# first category (in this order) that yields the LONGEST recent series.
CPA_PREFERENCE = ["CPA_F41001_X_410014",   # residential excl. communities
                  "CPA_F410010",           # residential buildings
                  "CPA_X_F41001",          # buildings (other codings)
                  None]                    # any remaining

# Greek monthly CCI: ELSTAT SPC23 (DKT60 timeseries export, manual
# download from statistics.gr; monthly rows only -- annual-average
# rows are interleaved and dropped). Column map fingerprint-verified
# against the Paper 2 series (r = 1.0000 on 2000-2024 overlap):
#   col 1 = General Index (used here), 2 = Concrete, 7 = Steel,
#   8 = PVC/plumbing, 16 = Fuel/energy.
ELSTAT_XLS = os.path.join(SCRIPT_DIR, "..", "data", "raw",
                          "elstat_spc23_2000_2026.xls")
ELSTAT_GENERAL_COL = 1


def _parse_time(t):
    if "-Q" in t:  # quarterly -> last month of quarter
        y, q = t.split("-Q")
        return pd.Timestamp(int(y), int(q) * 3, 1)
    return pd.Timestamp(t + "-01")


def fetch(dataset, geo, indic_bt, since):
    """One Eurostat series -> DataFrame [date, country, level, cpa].

    Queries WITHOUT a cpa2_1 filter (categories differ by country),
    decodes the multi-dimensional response generically, then keeps the
    preferred CPA category: the first in CPA_PREFERENCE whose series
    extends furthest; ties broken by series length.
    """
    params = dict(COMMON)
    params.update({"geo": geo, "indic_bt": indic_bt,
                   "sinceTimePeriod": since})
    r = requests.get(API + dataset, params=params, timeout=120)
    r.raise_for_status()
    d = r.json()
    if not d.get("value"):
        print(f"  {dataset} {geo} {indic_bt}: NO DATA")
        return pd.DataFrame(columns=["date", "country", "level", "cpa"])

    dim_ids = d["id"]
    sizes = d["size"]
    labels = {}
    for did in dim_ids:
        idx = d["dimension"][did]["category"]["index"]
        labels[did] = sorted(idx, key=lambda k: idx[k])

    rows = []
    for pos, val in d["value"].items():
        p = int(pos)
        coord = {}
        for did, sz in zip(reversed(dim_ids), reversed(sizes)):
            coord[did] = labels[did][p % sz]
            p //= sz
        rows.append({"date": _parse_time(coord["time"]),
                     "country": geo,
                     "level": float(val),
                     "cpa": coord.get("cpa2_1", "ALL")})
    df = pd.DataFrame(rows)

    def rank(cpa_series):
        end, n = cpa_series["date"].max(), len(cpa_series)
        return (end, n)

    best, best_key = None, None
    groups = {c: g for c, g in df.groupby("cpa")}
    ordered = [c for c in CPA_PREFERENCE if c in groups] + \
              [c for c in groups if c not in CPA_PREFERENCE]
    for c in ordered:
        g = groups[c]
        if best is None or rank(g) > rank(best):
            best, best_key = g, c
    df = best.sort_values("date").reset_index(drop=True)
    print(f"  {dataset} {geo} {indic_bt} [{best_key}]: {len(df)} obs "
          f"[{df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m}]")
    return df


def main():
    os.makedirs(RAW, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- Eurostat CCI downloader (v3)")
    print("=" * 70)

    print("\n[1] Monthly COST panel (7 countries):")
    frames = []
    for g in MONTHLY_COST_GEO:
        frames.append(fetch("sts_copi_m", g, "COST", "2000-01"))
        time.sleep(1)
    cost = pd.concat(frames, ignore_index=True)
    cost.to_csv(os.path.join(RAW, "eurostat_cci_monthly_COST.csv"),
                index=False)

    print("\n[2] Monthly PRC_PRR robustness set:")
    frames = []
    for g in MONTHLY_PRR_GEO:
        frames.append(fetch("sts_copi_m", g, "PRC_PRR", "2000-01"))
        time.sleep(1)
    prr = pd.concat(frames, ignore_index=True)
    prr.to_csv(os.path.join(RAW, "eurostat_cci_monthly_PRC_PRR.csv"),
               index=False)

    print("\n[3] Quarterly layer (DE, FR, EL):")
    frames = []
    for g in QUARTERLY_GEO:
        for ind in ["COST", "PRC_PRR"]:
            df = fetch("sts_copi_q", g, ind, "2000-Q1")
            if len(df):
                df["indicator"] = ind
                frames.append(df)
            time.sleep(1)
    qtr = pd.concat(frames, ignore_index=True)
    qtr.to_csv(os.path.join(RAW, "eurostat_cci_quarterly.csv"), index=False)

    print("\n[4] Greek monthly CCI from ELSTAT SPC23 (direct):")
    raw = pd.read_excel(ELSTAT_XLS, sheet_name=0, header=None)
    d = pd.to_datetime(raw.iloc[13:, 0], errors="coerce")
    keep = d.notna()
    lvl = raw.iloc[13:, ELSTAT_GENERAL_COL].astype(float)[keep.values]
    gr = pd.DataFrame({"date": d[keep].values, "country": "GR",
                       "level": lvl.values}).sort_values("date")
    gr.to_csv(os.path.join(RAW, "cci_GR_elstat.csv"), index=False)
    print(f"  GR (ELSTAT SPC23): {len(gr)} obs "
          f"[{gr['date'].min():%Y-%m} -> {gr['date'].max():%Y-%m}]")
    if gr["date"].max() < pd.Timestamp("2026-01-01"):
        print("  !! GR series ends before 2026 -- refresh the ELSTAT "
              "SPC23 download before submission (manual, statistics.gr).")

    print("\nDONE -- 01_eurostat_cci.py")


if __name__ == "__main__":
    main()
