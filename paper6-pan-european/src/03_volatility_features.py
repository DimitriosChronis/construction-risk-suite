"""
03_volatility_features.py
=========================
Rolling volatility and crisis-regime labels for every panel country.

Two label families (leak-free by construction):
  * ENDOGENOUS labels: rolling 6M sigma vs the EXPANDING-WINDOW
    (point-in-time) 75th percentile of that country's own sigma
    history up to month t. No future information enters any label.
    A 36-month burn-in precedes the first label.
  * EXOGENOUS windows: the four pre-registered EU-wide crisis
    episodes (GFC, sovereign, COVID, Ukraine), identical for all
    countries and independent of the data.

Inputs : data/processed/cci_panel_{primary,extended,cost}.csv
Outputs: data/processed/cci_features_{primary,extended,cost}.csv
         (date, country, level, log_return, sigma6,
          crisis_endog, crisis_exog, exog_episode)
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")

VOL_WINDOW = 6
CRISIS_PCT = 0.75
MIN_HIST = 36              # months of sigma history before first label

EXOG_WINDOWS = {           # pre-registered, EU-wide, data-independent
    "GFC":       ("2008-09-01", "2009-12-31"),
    "SOVEREIGN": ("2011-07-01", "2012-12-31"),
    "COVID":     ("2020-03-01", "2020-12-31"),
    "UKRAINE":   ("2022-02-01", "2022-12-31"),
}


def point_in_time_labels(sigma):
    """Expanding-percentile crisis labels; NaN during burn-in."""
    lab = np.full(len(sigma), np.nan)
    vals = sigma.values
    for i in range(len(vals)):
        hist = vals[: i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < MIN_HIST or np.isnan(vals[i]):
            continue
        thr = np.quantile(hist, CRISIS_PCT)
        lab[i] = 1.0 if vals[i] > thr else 0.0
    return lab


def process(panel_name):
    path = os.path.join(PROC, f"cci_panel_{panel_name}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    out = []
    for c, g in df.groupby("country"):
        g = g.sort_values("date").copy()
        g["sigma6"] = g["log_return"].rolling(VOL_WINDOW).std()
        g["crisis_endog"] = point_in_time_labels(g["sigma6"])
        g["crisis_exog"] = 0
        g["exog_episode"] = ""
        for name, (a, b) in EXOG_WINDOWS.items():
            m = (g["date"] >= a) & (g["date"] <= b)
            g.loc[m, "crisis_exog"] = 1
            g.loc[m, "exog_episode"] = name
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(PROC, f"cci_features_{panel_name}.csv"),
               index=False)
    lab = res.dropna(subset=["crisis_endog"])
    share = lab.groupby("country")["crisis_endog"].mean()
    print(f"[{panel_name}] endog crisis share by country "
          f"(target ~{1-CRISIS_PCT:.0%}):")
    print("  " + "  ".join(f"{c}:{v:.2f}" for c, v in share.items()))
    exog_share = res["crisis_exog"].mean()
    print(f"  exogenous crisis months: {exog_share:.1%} of panel")
    return res


def main():
    print("=" * 70)
    print("Paper 6 -- Volatility features + leak-free labels")
    print("=" * 70)
    for name in ["primary", "extended", "cost"]:
        process(name)
    print("\nDONE -- 03_volatility_features.py")


if __name__ == "__main__":
    main()
