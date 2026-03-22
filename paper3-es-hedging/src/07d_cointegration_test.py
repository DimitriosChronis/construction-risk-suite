"""
07d_cointegration_test.py
Fix 2: Engle-Granger Cointegration Test for Hedge Pairs

Tests whether GR-US price LEVELS are cointegrated (long-run equilibrium).
If NOT cointegrated => no long-run mean-reversion => basis risk is structural,
not temporary => explains why HE is low.

Also adds ADF stationarity test on log-returns (should be stationary).

Input:  data/raw/elstat_data.xlsx, data/raw/global_commodities_monthly.csv
        data/processed/aligned_log_returns.csv
Output: results/tables/c5d_cointegration.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",         "Steel"),
    ("US_Cement_PPI", "GR_Concrete",      "Cement"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",   "Fuel"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",     "PVC"),
    ("US_Brent",      "GR_General_Index", "General"),
]

CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Log-returns data: {df.shape}")

    # Reconstruct price levels from log-returns for cointegration test
    # (cointegration requires I(1) price levels, not I(0) returns)
    prices = np.exp(df.cumsum())  # cumulative log-returns -> price index
    crisis_prices = prices.loc[CRISIS_START:CRISIS_END]

    rows = []
    for us_col, gr_col, label in PAIRS:
        if us_col not in df.columns or gr_col not in df.columns:
            print(f"  SKIP: {label}")
            continue

        print(f"\n--- {label} ({us_col} vs {gr_col}) ---")

        # 1. ADF on log-returns (should be stationary = I(0))
        adf_us = adfuller(df[us_col].dropna(), maxlag=12, autolag="AIC")
        adf_gr = adfuller(df[gr_col].dropna(), maxlag=12, autolag="AIC")
        print(f"  ADF log-returns: US p={adf_us[1]:.4f} "
              f"{'I(0)' if adf_us[1]<0.05 else 'NOT stationary'}  "
              f"GR p={adf_gr[1]:.4f} "
              f"{'I(0)' if adf_gr[1]<0.05 else 'NOT stationary'}")

        # 2. ADF on price levels (should be non-stationary = I(1))
        adf_us_lvl = adfuller(prices[us_col].dropna(), maxlag=12, autolag="AIC")
        adf_gr_lvl = adfuller(prices[gr_col].dropna(), maxlag=12, autolag="AIC")
        print(f"  ADF price levels: US p={adf_us_lvl[1]:.4f} "
              f"{'I(1)' if adf_us_lvl[1]>0.05 else 'I(0)'}  "
              f"GR p={adf_gr_lvl[1]:.4f} "
              f"{'I(1)' if adf_gr_lvl[1]>0.05 else 'I(0)'}")

        # 3. Engle-Granger cointegration test (full period)
        try:
            coint_stat, coint_p, crit_vals = coint(
                prices[us_col].dropna().values,
                prices[gr_col].dropna().values,
                trend="c", maxlag=12, autolag="AIC"
            )
            coint_pass = coint_p < 0.05
        except Exception as e:
            print(f"  Cointegration error: {e}")
            coint_stat, coint_p, coint_pass = np.nan, np.nan, False

        print(f"  Engle-Granger (full): stat={coint_stat:.3f}, "
              f"p={coint_p:.4f}, "
              f"{'COINTEGRATED' if coint_pass else 'NOT cointegrated'}")

        # 4. Crisis-period cointegration
        try:
            coint_c_stat, coint_c_p, _ = coint(
                crisis_prices[us_col].dropna().values,
                crisis_prices[gr_col].dropna().values,
                trend="c", maxlag=6, autolag="AIC"
            )
            coint_c_pass = coint_c_p < 0.05
        except Exception:
            coint_c_stat, coint_c_p, coint_c_pass = np.nan, np.nan, False

        print(f"  Engle-Granger (crisis): stat={coint_c_stat:.3f}, "
              f"p={coint_c_p:.4f}, "
              f"{'COINTEGRATED' if coint_c_pass else 'NOT cointegrated'}")

        rows.append({
            "Pair": label,
            "US_series": us_col,
            "GR_series": gr_col,
            "ADF_US_returns_p": round(adf_us[1], 4),
            "ADF_GR_returns_p": round(adf_gr[1], 4),
            "Returns_stationary": adf_us[1] < 0.05 and adf_gr[1] < 0.05,
            "ADF_US_levels_p": round(adf_us_lvl[1], 4),
            "ADF_GR_levels_p": round(adf_gr_lvl[1], 4),
            "EG_full_stat": round(coint_stat, 3),
            "EG_full_p": round(coint_p, 4),
            "Cointegrated_full": coint_pass,
            "EG_crisis_stat": round(coint_c_stat, 3) if not np.isnan(coint_c_stat) else "NA",
            "EG_crisis_p": round(coint_c_p, 4) if not np.isnan(coint_c_p) else "NA",
            "Cointegrated_crisis": coint_c_pass,
        })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_TAB, "c5d_cointegration.csv"), index=False)
    print(f"\nSaved: c5d_cointegration.csv")
    print(res_df[["Pair", "EG_full_p", "Cointegrated_full",
                   "EG_crisis_p", "Cointegrated_crisis"]].to_string(index=False))

    # Summary
    n_coint = sum(1 for r in rows if r["Cointegrated_full"])
    print(f"\nSummary: {n_coint}/{len(rows)} pairs cointegrated in full period")
    if n_coint == 0:
        print("=> NO long-run equilibrium between GR and US prices")
        print("=> Basis risk is STRUCTURAL, not temporary")
        print("=> This explains the low hedge effectiveness")


if __name__ == "__main__":
    main()
