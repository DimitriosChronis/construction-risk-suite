"""
08_cross_country_panel.py
=========================
H4: heterogeneous transmission of US material-price volatility into
European construction-cost growth (risk inequality).

  dln(CCI)_{c,t} = a_c + g_t + b1 * vol_{t-1}
                   + b2 * vol_{t-1} x 1[Mediterranean] + e_{c,t}

vol_t = composite US material-price volatility: mean of the five
6-month rolling PPI-return std devs (Brent, Steel, Cement, Fuel,
PVC), standardised. Downloaded FRESH from FRED (BLS series are still
updated, unlike the frozen OECD family), so the panel runs through
2026 for the seven Eurostat countries. Month fixed effects absorb
common shocks; b1 is therefore identified from the RELATIVE
transmission and a no-time-FE variant reports the aggregate effect.
SEs clustered by country (8 clusters -- caveat reported; HC3 shown
as robustness).

Inputs : data/processed/cci_features_primary.csv, FRED (live)
Outputs: data/raw/us_ppi_monthly.csv
         results/tables/table_panel_interaction.csv
         results/tables/table_h4_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

FRED_SERIES = {"Brent": "DCOILBRENTEU", "Steel": "WPU101",
               "Cement": "WPU1321", "Fuel": "WPU0553", "PVC": "WPU0721"}
MED = {"GR", "ES", "IT", "PT"}
VOL_W = 6


def get_us_vol():
    import datetime as dt
    frames = {}
    for name, sid in FRED_SERIES.items():
        s = web.DataReader(sid, "fred", dt.datetime(1999, 1, 1),
                           dt.datetime(2026, 12, 31)).iloc[:, 0]
        s = s.resample("MS").mean()
        frames[name] = np.log(s).diff()
    r = pd.DataFrame(frames)
    r.to_csv(os.path.join(RAW, "us_ppi_monthly.csv"))
    vol = r.rolling(VOL_W).std()
    z = (vol - vol.mean()) / vol.std()
    comp = z.mean(axis=1).rename("us_vol")
    print(f"US composite vol: {comp.dropna().index.min():%Y-%m} -> "
          f"{comp.dropna().index.max():%Y-%m}")
    return comp


def main():
    os.makedirs(TAB, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- H4 panel: US vol -> EU CCI growth, MED interaction")
    print("=" * 70)

    vol = get_us_vol()
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    df = feat[["date", "country", "log_return"]].dropna().copy()
    df["dlncci_pct"] = 100 * df["log_return"]
    df["vol_l1"] = df["date"].map(vol.shift(1))
    df["med"] = df["country"].isin(MED).astype(int)
    df["vol_x_med"] = df["vol_l1"] * df["med"]
    df = df.dropna(subset=["vol_l1"])
    print(f"panel: {df['country'].nunique()} countries x "
          f"{df['date'].nunique()} months = {len(df)} obs "
          f"[{df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m}]")

    rows = []
    # (1) aggregate effect, country FE only
    m1 = smf.ols("dlncci_pct ~ vol_l1 + C(country)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})
    # (2) interaction, country FE
    m2 = smf.ols("dlncci_pct ~ vol_l1 + vol_x_med + C(country)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})
    # (3) interaction, country + month FE (b1 absorbed; b2 identified)
    m3 = smf.ols("dlncci_pct ~ vol_x_med + C(country) + C(date)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})
    # (4) HC3 robustness of (2)
    m4 = smf.ols("dlncci_pct ~ vol_l1 + vol_x_med + C(country)", df).fit(
        cov_type="HC3")

    for name, m, terms in [
        ("M1_countryFE", m1, ["vol_l1"]),
        ("M2_interaction", m2, ["vol_l1", "vol_x_med"]),
        ("M3_timeFE", m3, ["vol_x_med"]),
        ("M4_HC3", m4, ["vol_l1", "vol_x_med"]),
    ]:
        for t in terms:
            rows.append({"model": name, "term": t,
                         "beta_pp_per_sigma": round(m.params[t], 4),
                         "se": round(m.bse[t], 4),
                         "p_value": float(f"{m.pvalues[t]:.3e}"),
                         "n": int(m.nobs)})
            print(f"  {name:15s} {t:10s} b={m.params[t]:+.4f} "
                  f"(se {m.bse[t]:.4f}, p={m.pvalues[t]:.3e})")

    pd.DataFrame(rows).to_csv(
        os.path.join(TAB, "table_panel_interaction.csv"), index=False)

    b1 = m2.params["vol_l1"]; b2 = m2.params["vol_x_med"]
    pd.DataFrame([{
        "beta1_core_ppsigma": round(b1, 4),
        "beta2_med_extra_ppsigma": round(b2, 4),
        "med_to_core_ratio": round((b1 + b2) / b1, 3) if b1 != 0 else np.nan,
        "beta2_p_cluster": float(f"{m2.pvalues['vol_x_med']:.3e}"),
        "beta2_p_timeFE": float(f"{m3.pvalues['vol_x_med']:.3e}"),
        "n_obs": int(m2.nobs), "n_countries": df["country"].nunique(),
        "note": "8 clusters; cluster-robust SEs conservative caveat",
    }]).to_csv(os.path.join(TAB, "table_h4_summary.csv"), index=False)

    print("\nDONE -- 08_cross_country_panel.py")


if __name__ == "__main__":
    main()
