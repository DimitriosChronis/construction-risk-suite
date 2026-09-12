"""
08b_h4_wildboot.py
==================
G5a: wild cluster bootstrap (Rademacher, clusters = countries,
B = 4999) for the H4 interaction coefficient beta2 -- the honest
inference given only 8 clusters, where analytic clustered SEs are
unreliable. Restricted (null-imposed) bootstrap of the t-statistic
(Cameron-Gelbach-Miller).

Inputs : data/processed/cci_features_primary.csv,
         data/raw/us_ppi_monthly.csv
Outputs: results/tables/table_h4_wildboot.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
RAW = os.path.join(SCRIPT_DIR, "..", "data", "raw")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

B = 4999
SEED = 42
MED = {"GR", "ES", "IT", "PT"}
VOL_W = 6


def build_panel():
    r = pd.read_csv(os.path.join(RAW, "us_ppi_monthly.csv"),
                    index_col=0, parse_dates=True)
    vol = r.rolling(VOL_W).std()
    z = (vol - vol.mean()) / vol.std()
    comp = z.mean(axis=1)
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    df = feat[["date", "country", "log_return"]].dropna().copy()
    df["dlncci_pct"] = 100 * df["log_return"]
    df["vol_l1"] = df["date"].map(comp.shift(1))
    df["med"] = df["country"].isin(MED).astype(int)
    df["vol_x_med"] = df["vol_l1"] * df["med"]
    return df.dropna(subset=["vol_l1"]).reset_index(drop=True)


def tstat_b2(df):
    m = smf.ols("dlncci_pct ~ vol_l1 + vol_x_med + C(country)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"]})
    return m.params["vol_x_med"], m.params["vol_x_med"] / m.bse["vol_x_med"]


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- G5a wild cluster bootstrap for H4 beta2")
    print("=" * 70)
    df = build_panel()
    b2, t_obs = tstat_b2(df)
    print(f"beta2 = {b2:+.4f}, cluster t = {t_obs:.3f}")

    # Null-imposed model (beta2 = 0)
    m0 = smf.ols("dlncci_pct ~ vol_l1 + C(country)", df).fit()
    resid0 = m0.resid.values
    fitted0 = m0.fittedvalues.values
    clusters = df["country"].values
    uniq = np.unique(clusters)

    exceed = 0
    for _ in range(B):
        w = dict(zip(uniq, rng.choice([-1.0, 1.0], size=len(uniq))))
        yb = fitted0 + resid0 * np.array([w[c] for c in clusters])
        db = df.copy()
        db["dlncci_pct"] = yb
        _, t_b = tstat_b2(db)
        if abs(t_b) >= abs(t_obs):
            exceed += 1
    p = (exceed + 1) / (B + 1)
    print(f"wild-cluster bootstrap p (two-sided) = {p:.4f}  (B={B})")

    pd.DataFrame([{"beta2": round(b2, 4), "t_cluster": round(t_obs, 3),
                   "wild_boot_p": round(p, 4), "B": B,
                   "n_clusters": len(uniq)}]).to_csv(
        os.path.join(TAB, "table_h4_wildboot.csv"), index=False)
    print("\nDONE -- 08b_h4_wildboot.py")


if __name__ == "__main__":
    main()
