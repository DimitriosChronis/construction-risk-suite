"""
04e_cholesky_sensitivity.py
===========================
Revision response to R1-M3 (Cholesky ordering sensitivity).

Methodology:
  * For each of the 5 matched bivariate VARs, estimate IRF under
    two orderings: (i) US first (the published specification),
    (ii) GR first (reversed).
  * Compute the cumulative 12-month response of the Greek series
    to a one-standard-deviation US shock under each ordering.
  * Report the % difference. If the difference is small, the
    Cholesky ordering does not drive the published results.
  * Also estimate the analogous full-system IRF with both
    orderings within the 10-variable VAR.
  * The vine-copula evidence (US Fuel PPI as hub, vine-strength
    0.464) provides INDEPENDENT theoretical justification for the
    US-first ordering -- this is the inferential link to the
    vine layer that R2-a requested.

Inputs:
    data/processed/aligned_log_returns.csv
Outputs:
    results/tables/c2e_cholesky_sensitivity.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                          "aligned_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

H_IRF      = 12
MAX_LAG    = 6

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",         "Steel"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",   "Fuel/Energy"),
    ("US_Cement_PPI", "GR_Concrete",      "Cement/Concrete"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",     "PVC/Plastic"),
    ("US_Brent",      "GR_General_Index", "Brent/General"),
]


def cumulative_us_to_gr_response(res, us_var, gr_var, h):
    """Cumulative h-period orthogonalised IRF: response of gr_var to a
    1-SD shock in us_var, using the current Cholesky ordering."""
    irf = res.irf(h)
    # orth_irfs shape: (h+1, n_vars, n_vars) -> [horizon, response, shock]
    us_idx = res.names.index(us_var)
    gr_idx = res.names.index(gr_var)
    series = irf.orth_irfs[:, gr_idx, us_idx]
    return float(np.cumsum(series)[-1])


def fit_with_ordering(df, ordering, lag):
    Z = df[ordering].dropna()
    res = VAR(Z).fit(maxlags=lag, ic=None)
    return res


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print("=" * 70)
    print("Paper 2 -- Revision: Cholesky ordering sensitivity (04e)")
    print("=" * 70)
    print(f"  Data: {df.shape}")

    rows = []
    for us, gr, label in PAIRS:
        sub = df[[us, gr]].dropna()
        # Pick a common lag via AIC on the original ordering
        try:
            p_star = max(1, int(VAR(sub[[us, gr]]).select_order(MAX_LAG).aic))
        except Exception:
            p_star = 2

        # Original ordering: [US, GR] -> US-first
        res_us = fit_with_ordering(sub, [us, gr], p_star)
        cum_us = cumulative_us_to_gr_response(res_us, us, gr, H_IRF)

        # Reversed ordering: [GR, US]
        res_gr = fit_with_ordering(sub, [gr, us], p_star)
        cum_gr = cumulative_us_to_gr_response(res_gr, us, gr, H_IRF)

        if abs(cum_us) > 1e-9:
            pct_diff = 100.0 * (cum_gr - cum_us) / abs(cum_us)
        else:
            pct_diff = np.nan

        rows.append({
            "pair":             label,
            "US":               us,
            "GR":               gr,
            "p_star":           p_star,
            "cum_IRF_US_first": round(cum_us, 5),
            "cum_IRF_GR_first": round(cum_gr, 5),
            "abs_diff":         round(cum_gr - cum_us, 5),
            "pct_diff":         round(pct_diff, 2),
        })
        print(f"  {label:20s} p={p_star}  "
              f"US-first={cum_us:+.4f}  GR-first={cum_gr:+.4f}  "
              f"diff={pct_diff:+.1f}%")

    biv_df = pd.DataFrame(rows)
    biv_df.to_csv(os.path.join(OUT_TAB,
                               "c2e_cholesky_sensitivity.csv"),
                  index=False)

    # -------------------------------------------------------------------------
    # System-level cross-check
    # -------------------------------------------------------------------------
    print("\n[system] 10-variable VAR cross-check:")
    full_us_first = ["US_Brent", "US_Steel_PPI", "US_Cement_PPI",
                     "US_Fuel_PPI", "US_PVC_PPI",
                     "GR_General_Index", "GR_Concrete", "GR_Steel",
                     "GR_Fuel_Energy", "GR_PVC_Pipes"]
    full_gr_first = full_us_first[5:] + full_us_first[:5]

    full_sub = df[full_us_first].dropna()
    p_full = max(1, int(VAR(full_sub).select_order(MAX_LAG).aic))
    print(f"  full-VAR p* = {p_full}")

    res_uf = fit_with_ordering(df, full_us_first, p_full)
    res_gf = fit_with_ordering(df, full_gr_first, p_full)

    sys_rows = []
    for us, gr, label in PAIRS:
        c_uf = cumulative_us_to_gr_response(res_uf, us, gr, H_IRF)
        c_gf = cumulative_us_to_gr_response(res_gf, us, gr, H_IRF)
        sys_rows.append({
            "pair":      label,
            "US-first":  round(c_uf, 5),
            "GR-first":  round(c_gf, 5),
            "pct_diff":  round(100.0 * (c_gf - c_uf) / abs(c_uf), 2)
                          if abs(c_uf) > 1e-9 else np.nan,
        })
    sys_df = pd.DataFrame(sys_rows)
    sys_df.to_csv(os.path.join(OUT_TAB,
                               "c2e_cholesky_system.csv"), index=False)
    print(sys_df)

    print("\n" + "=" * 70)
    print("DONE -- 04e_cholesky_sensitivity.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
