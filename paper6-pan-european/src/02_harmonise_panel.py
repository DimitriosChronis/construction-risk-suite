"""
02_harmonise_panel.py
=====================
Builds the primary monthly panel (v3): eight euro-area countries,
2000-01 -> latest common coverage, log-returns, coverage report.

Primary panel:  GR (ELSTAT cost index), ES, IT, PT (Mediterranean)
                NL, IE, FI, LT          (Northern)
Indicator:      PRC_PRR for the seven Eurostat countries.
Extended set:   +NO, +PL (non-euro robustness).
COST subset:    ES, PT, NL, LT, NO (indicator robustness).

Inputs  (data/raw/): eurostat_cci_monthly_PRC_PRR.csv,
                     eurostat_cci_monthly_COST.csv, cci_GR_elstat.csv
Outputs (data/processed/):
    cci_panel_primary.csv    (long: date,country,level,log_return)
    cci_panel_extended.csv   (primary + NO + PL)
    cci_panel_cost.csv       (COST-indicator subset)
    panel_coverage.csv       (per-country span/gap report)
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")

PRIMARY = ["GR", "ES", "IT", "PT", "NL", "IE", "FI", "LT"]
EXTENDED_EXTRA = ["NO", "PL"]
COST_SET = ["ES", "PT", "NL", "LT", "NO"]
START = "2000-01-01"
MEDITERRANEAN = {"GR", "ES", "IT", "PT"}


def add_returns(df):
    out = []
    for c, g in df.groupby("country"):
        g = g.sort_values("date").copy()
        g["log_return"] = np.log(g["level"]).diff()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def coverage(df, name):
    rows = []
    for c, g in df.groupby("country"):
        g = g.sort_values("date")
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        rows.append({
            "panel": name, "country": c,
            "start": f"{g['date'].min():%Y-%m}",
            "end": f"{g['date'].max():%Y-%m}",
            "n_obs": len(g),
            "n_missing_internal": len(full) - len(g),
            "bloc": ("MED" if c in MEDITERRANEAN else "NORTH"),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(PROC, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- Harmonise panel (v3)")
    print("=" * 70)

    prr = pd.read_csv(os.path.join(RAW, "eurostat_cci_monthly_PRC_PRR.csv"),
                      parse_dates=["date"])
    cost = pd.read_csv(os.path.join(RAW, "eurostat_cci_monthly_COST.csv"),
                       parse_dates=["date"])
    gr = pd.read_csv(os.path.join(RAW, "cci_GR_elstat.csv"),
                     parse_dates=["date"])

    prr = prr[prr["date"] >= START][["date", "country", "level"]]
    cost = cost[cost["date"] >= START][["date", "country", "level"]]
    gr = gr[gr["date"] >= START][["date", "country", "level"]]

    primary = pd.concat(
        [gr[gr["country"] == "GR"],
         prr[prr["country"].isin([c for c in PRIMARY if c != "GR"])]],
        ignore_index=True)
    primary = add_returns(primary)
    primary.to_csv(os.path.join(PROC, "cci_panel_primary.csv"), index=False)

    extended = pd.concat(
        [primary[["date", "country", "level", "log_return"]],
         add_returns(prr[prr["country"].isin(EXTENDED_EXTRA)])],
        ignore_index=True)
    extended.to_csv(os.path.join(PROC, "cci_panel_extended.csv"), index=False)

    cost_panel = add_returns(cost[cost["country"].isin(COST_SET)])
    cost_panel.to_csv(os.path.join(PROC, "cci_panel_cost.csv"), index=False)

    cov = pd.concat([coverage(primary, "primary"),
                     coverage(extended, "extended"),
                     coverage(cost_panel, "cost")], ignore_index=True)
    cov.to_csv(os.path.join(PROC, "panel_coverage.csv"), index=False)
    print(cov[cov["panel"] == "primary"].to_string(index=False))

    # Sanity: return magnitudes plausible (monthly |r| rarely > 15%)
    big = primary[primary["log_return"].abs() > 0.15]
    print(f"\n|log-return| > 15%: {len(big)} obs")
    if len(big):
        print(big.to_string(index=False))
    print("\nDONE -- 02_harmonise_panel.py")


if __name__ == "__main__":
    main()
