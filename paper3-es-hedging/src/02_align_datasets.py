"""
02_align_datasets.py
Merge ELSTAT Greek indices with US FRED global commodity indices.
Compute monthly log-returns. Align on common date range (2000-2024).

Input:
    data/raw/elstat_data.xlsx                (ELSTAT construction cost indices)
    data/raw/global_commodities_monthly.csv  (from script 01)

Output:
    data/processed/aligned_log_returns.csv   All 10 series log-returns
    data/processed/elstat_log_returns.csv    Greek only (5 series)
    data/processed/global_log_returns.csv    Global only (5 series)
"""

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import os

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(__file__)
ELSTAT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "elstat_data.xlsx")
GLOBAL_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw",
                            "global_commodities_monthly.csv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "processed")

# ELSTAT Excel layout: header at row 13 (skiprows=12)
ELSTAT_COL_MAP = {0: "Date", 1: "General_Index", 2: "Concrete",
                  7: "Steel", 16: "Fuel_Energy", 8: "PVC_Pipes"}


def load_elstat(path):
    """Load ELSTAT price levels from raw Excel, compute log-returns."""
    df = pd.read_excel(path, skiprows=12, engine="openpyxl")
    rename = {df.columns[idx]: name for idx, name in ELSTAT_COL_MAP.items()}
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.loc["2000-01-01":"2024-12-31"]
    target = [c for c in ELSTAT_COL_MAP.values() if c != "Date"]
    df = df[[c for c in target if c in df.columns]]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    # Normalize to month-start
    df.index = df.index.to_period("M").to_timestamp()
    log_ret = np.log(df / df.shift(1)).dropna()
    log_ret.columns = [f"GR_{c}" for c in log_ret.columns]
    return log_ret


def load_global(path):
    """Load FRED global commodity levels, compute log-returns."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.to_period("M").to_timestamp()
    df = df[~df.index.duplicated(keep="last")]
    log_ret = np.log(df / df.shift(1)).dropna()
    log_ret.columns = [f"US_{c}" for c in log_ret.columns]
    return log_ret


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    elstat  = load_elstat(ELSTAT_PATH)
    global_ = load_global(GLOBAL_PATH)

    print(f"ELSTAT:  {elstat.shape}  | {elstat.index[0].date()} to {elstat.index[-1].date()}")
    print(f"Global:  {global_.shape} | {global_.index[0].date()} to {global_.index[-1].date()}")

    # Inner join on common months
    aligned = pd.concat([elstat, global_], axis=1, join="inner").dropna()
    print(f"\nAligned: {aligned.shape} | {aligned.index[0].date()} to {aligned.index[-1].date()}")
    print(f"Columns: {list(aligned.columns)}")

    # Save all three datasets
    aligned.to_csv(os.path.join(OUT_DIR, "aligned_log_returns.csv"))
    elstat_aligned = aligned[[c for c in aligned.columns if c.startswith("GR_")]]
    global_aligned = aligned[[c for c in aligned.columns if c.startswith("US_")]]
    elstat_aligned.to_csv(os.path.join(OUT_DIR, "elstat_log_returns.csv"))
    global_aligned.to_csv(os.path.join(OUT_DIR, "global_log_returns.csv"))

    # Cross-market Kendall tau (matching pairs)
    pairs = [
        ("GR_Steel",       "US_Steel_PPI"),
        ("GR_Fuel_Energy", "US_Fuel_PPI"),
        ("GR_Concrete",    "US_Cement_PPI"),
        ("GR_PVC_Pipes",   "US_PVC_PPI"),
        ("GR_General_Index","US_Brent"),
    ]
    print("\nCross-market Kendall tau (GR vs US):")
    for gr, us in pairs:
        if gr in aligned.columns and us in aligned.columns:
            tau, p = kendalltau(aligned[gr], aligned[us])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {gr:22s} vs {us:16s}  tau={tau:+.3f}  p={p:.4f} {sig}")

    print("\nSaved: aligned_log_returns.csv, elstat_log_returns.csv, global_log_returns.csv")


if __name__ == "__main__":
    main()
