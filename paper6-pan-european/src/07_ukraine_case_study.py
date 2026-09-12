"""
07_ukraine_case_study.py
========================
B5: the 2022 Ukraine energy shock across the 8-country panel --
the episode-level illustration of H1/H3.

Per country: sigma6 peak month within 2022-02..2023-06, shock ratio
(mean sigma6 in the Ukraine window vs calm 2019), months from
invasion (2022-02) to sigma6 peak. Panel: mean lambda_U (w=24)
trajectory 2021-01..2023-12 with its full-sample percentile.

Inputs : data/processed/cci_features_primary.csv,
         data/processed/lambdaU_pairs_w24.csv
Outputs: results/tables/table_ukraine_case.csv
         results/tables/table_ukraine_network.csv
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

WINDOW = ("2022-02-01", "2023-06-30")
BASELINE = ("2019-01-01", "2019-12-31")
INVASION = pd.Timestamp("2022-02-01")


def main():
    os.makedirs(TAB, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- B5 Ukraine 2022 case study")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    rows = []
    for c, g in feat.groupby("country"):
        g = g.set_index("date").sort_index()
        win = g.loc[WINDOW[0]:WINDOW[1], "sigma6"].dropna()
        base = g.loc[BASELINE[0]:BASELINE[1], "sigma6"].dropna()
        if win.empty or base.empty:
            continue
        peak_m = win.idxmax()
        rows.append({
            "country": c,
            "sigma6_2019_mean": round(base.mean(), 5),
            "sigma6_ukr_mean": round(win.mean(), 5),
            "shock_ratio": round(win.mean() / base.mean(), 2),
            "sigma6_peak": f"{peak_m:%Y-%m}",
            "months_invasion_to_peak":
                (peak_m.year - INVASION.year) * 12
                + peak_m.month - INVASION.month,
            "endog_crisis_share_ukr": round(
                g.loc[WINDOW[0]:WINDOW[1], "crisis_endog"].mean(), 2),
        })
    case = pd.DataFrame(rows).sort_values("shock_ratio", ascending=False)
    case.to_csv(os.path.join(TAB, "table_ukraine_case.csv"), index=False)
    print(case.to_string(index=False))

    ts = pd.read_csv(os.path.join(PROC, "lambdaU_pairs_w24.csv"),
                     index_col=0, parse_dates=True)
    pm = ts.mean(axis=1).dropna()
    seg = pm.loc["2021-01-01":"2023-12-31"]
    pct = pm.rank(pct=True)
    net = pd.DataFrame({
        "panel_mean_lambdaU": seg.round(4),
        "full_sample_percentile": pct.loc[seg.index].round(3),
    })
    net.to_csv(os.path.join(TAB, "table_ukraine_network.csv"))
    peak = seg.idxmax()
    print(f"\npanel-mean lambda_U peak in episode: {peak:%Y-%m} "
          f"({seg.max():.3f}; {pct.loc[peak]:.0%} percentile of "
          f"full sample)")
    print("\nDONE -- 07_ukraine_case_study.py")


if __name__ == "__main__":
    main()
