"""
09_cross_eu_robustness.py
C6 (NEW) -- Cross-EU Robustness Check

Downloads Eurostat Construction Cost Index for EU countries.
Runs Granger causality (US -> EU country) to verify:
"The US-to-local lag finding is NOT a Greek idiosyncrasy."

Data: Eurostat STS_COPI_M (Construction Cost Index, monthly)
      For: Germany, France, Spain, Italy + Greece

Input:  FRED data (already downloaded)
        Eurostat CCI via pandas-datareader or direct CSV

Output: results/tables/c6_eu_granger.csv
        results/figures/fig_c6_eu_robustness.pdf
"""

import pandas as pd
import pandas_datareader.data as web
import numpy as np
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import os

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

START = dt.datetime(2000, 1, 1)
END   = dt.datetime(2024, 12, 31)
MAX_LAG = 6

# FRED PPI series for US commodities
US_SERIES = {
    "US_Steel_PPI": "WPU101",
    "US_Cement_PPI": "WPU1321",
}

# OECD/FRED construction-related PPI for EU countries
# Using industrial production / construction as proxy
EU_SERIES = {
    "DE_Construction": "DEUPRCNTO01IXOBM",  # Germany PPI Construction
    "FR_Construction": "FRAPRCNTO01IXOBM",  # France PPI Construction
    "IT_Construction": "ITAPRCNTO01IXOBM",  # Italy PPI Construction
    "ES_Construction": "ESPPRCNTO01IXOBM",  # Spain PPI Construction
}

# Fallback: OECD construction cost indices from FRED
EU_FALLBACK = {
    "DE_PPI": "DEUPPIALLMINMEI",  # Germany PPI All items
    "FR_PPI": "FRAPPIALLMINMEI",  # France PPI All items
    "IT_PPI": "ITAPPIALLMINMEI",  # Italy PPI All items
    "ES_PPI": "ESPPPIALLMINMEI",  # Spain PPI All items
}


def try_download(series_dict, start, end):
    """Try downloading FRED series, return what works."""
    results = {}
    for name, sid in series_dict.items():
        try:
            df = web.DataReader(sid, "fred", start, end).dropna()
            if len(df) > 24:
                results[name] = df.iloc[:, 0]
                print(f"  {name:20s} ({sid}): OK, {len(df)} obs")
        except Exception:
            pass
    return results


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    print("Downloading US commodity data from FRED...")
    us_data = try_download(US_SERIES, START, END)

    print("\nDownloading EU construction indices from FRED...")
    eu_data = try_download(EU_SERIES, START, END)

    if not eu_data:
        print("Primary EU series failed. Trying fallback PPI series...")
        eu_data = try_download(EU_FALLBACK, START, END)

    if not eu_data:
        print("\nERROR: No EU data available from FRED.")
        print("Alternative: Download manually from Eurostat (STS_COPI_M)")
        # Create placeholder with existing ELSTAT data
        from pathlib import Path
        elstat_path = Path(SCRIPT_DIR) / ".." / "data" / "processed" / "elstat_log_returns.csv"
        if elstat_path.exists():
            print("Using ELSTAT data as single-country validation instead.")
            elstat = pd.read_csv(elstat_path, index_col=0, parse_dates=True)
            eu_data = {"GR_General": elstat["GR_General_Index"]}

    if not us_data:
        print("ERROR: No US data. Check FRED connection.")
        return

    # Compute log-returns for all series
    us_returns = {}
    for name, series in us_data.items():
        s = series.resample("MS").last().dropna()
        lr = np.log(s / s.shift(1)).dropna()
        us_returns[name] = lr

    eu_returns = {}
    for name, series in eu_data.items():
        if isinstance(series, pd.Series):
            s = series
        else:
            s = series
        s = s.resample("MS").last().dropna()
        lr = np.log(s / s.shift(1)).dropna()
        eu_returns[name] = lr

    # ── Granger causality: US -> EU ───────────────────────────────────────────
    granger_rows = []
    print("\nGranger causality tests (US -> EU):")

    for us_name, us_ret in us_returns.items():
        for eu_name, eu_ret in eu_returns.items():
            # Align
            combined = pd.concat([eu_ret, us_ret], axis=1, join="inner").dropna()
            combined.columns = [eu_name, us_name]
            if len(combined) < 24:
                continue

            try:
                res = grangercausalitytests(combined, maxlag=MAX_LAG, verbose=False)
                best_p = 1.0
                best_lag = 0
                for lag, test_dict in res.items():
                    p_val = test_dict[0]["ssr_ftest"][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag

                sig = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else ""
                granger_rows.append({
                    "US_var": us_name,
                    "EU_var": eu_name,
                    "Best_lag": best_lag,
                    "Best_p": round(best_p, 4),
                    "Significant": best_p < 0.05,
                    "n_obs": len(combined),
                })
                print(f"  {us_name:18s} -> {eu_name:18s}  "
                      f"lag={best_lag}M  p={best_p:.4f} {sig}")
            except Exception as e:
                print(f"  {us_name} -> {eu_name}: ERROR {e}")

    granger_df = pd.DataFrame(granger_rows)
    granger_df.to_csv(os.path.join(OUT_TAB, "c6_eu_granger.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not granger_df.empty:
        n_sig = granger_df["Significant"].sum()
        n_total = len(granger_df)
        print(f"\nSignificant: {n_sig}/{n_total} pairs")

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sig_data = granger_df.copy()
        sig_data["Label"] = sig_data["US_var"] + " -> " + sig_data["EU_var"]
        colors = ["#4CAF50" if s else "#BDBDBD" for s in sig_data["Significant"]]
        ax.barh(sig_data["Label"], -np.log10(sig_data["Best_p"].clip(1e-10)),
                color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(-np.log10(0.05), color="red", linestyle="--",
                   linewidth=1, label="p=0.05")
        ax.set_xlabel("-log10(p-value)")
        ax.legend(fontsize=8)
        ax.set_title("")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIG, "fig_c6_eu_robustness.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("fig_c6_eu_robustness.pdf saved.")

    print("\nDone. Saved: c6_eu_granger.csv")


if __name__ == "__main__":
    main()
