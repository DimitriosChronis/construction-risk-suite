"""
01_global_data_download.py
Download global commodity price indices from FRED (St. Louis Fed).

Series:
    DCOILBRENTEU  - Brent Crude Oil spot price (USD/bbl, daily -> monthly)
    WPU101        - PPI Iron & Steel (monthly index)
    WPU1321       - PPI Cement (monthly index)
    WPU0721       - PPI Plastic Pipe (PVC proxy, monthly index)
    WPU0553       - PPI Fuel Oil (monthly index)

Output:
    data/raw/global_commodities_monthly.csv   (monthly price levels)
"""

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import os

# ── Config ────────────────────────────────────────────────────────────────────
START = dt.datetime(2000, 1, 1)
END   = dt.datetime(2024, 12, 31)

FRED_SERIES = {
    "Brent":       "DCOILBRENTEU",   # Daily -> resample monthly
    "Steel_PPI":   "WPU101",         # PPI Iron & Steel
    "Cement_PPI":  "WPU1321",        # PPI Cement
    "PVC_PPI":     "WPU0721",        # PPI Plastic Pipe
    "Fuel_PPI":    "WPU0553",        # PPI Fuel Oil
}

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "global_commodities_monthly.csv")


# ── Download ──────────────────────────────────────────────────────────────────
def download_fred_series(name, series_id, start, end):
    """Download a single FRED series."""
    print(f"  {name:12s} ({series_id})...", end=" ")
    try:
        df = web.DataReader(series_id, "fred", start, end)
        df = df.dropna()
        # Rename column
        df.columns = [name]
        print(f"OK, {len(df)} obs ({df.index[0].date()} to {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading global commodity data from FRED...\n")

    frames = []
    for name, sid in FRED_SERIES.items():
        df = download_fred_series(name, sid, START, END)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check network connection.")

    # Merge all series on date
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")

    # Resample Brent (daily) to monthly (last obs of month)
    merged = merged.resample("MS").last()
    merged.dropna(how="all", inplace=True)
    merged.index.name = "Date"

    # Forward-fill small gaps (max 2 months)
    merged = merged.ffill(limit=2)

    merged.to_csv(OUT_FILE)
    print(f"\nSaved: {OUT_FILE}")
    print(f"Shape: {merged.shape}")
    print(f"Period: {merged.index[0].date()} to {merged.index[-1].date()}")
    print(f"\nColumns: {list(merged.columns)}")
    print(merged.tail(6))
    print(f"\nMissing values:\n{merged.isna().sum()}")


if __name__ == "__main__":
    main()
