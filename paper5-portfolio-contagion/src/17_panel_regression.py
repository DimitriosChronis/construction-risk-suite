"""
Paper 5 -- B2b: Cross-country panel fixed-effects regression
=============================================================
Tests whether material-price volatility is a significant
determinant of construction-cost inflation in a panel of 17
EU/OECD countries (~3,000+ country-month observations), with
country and year fixed effects.

Specification:
  Δlog(CCI)_{c,t} = α_c + γ_t + β · vol_{t} + ε_{c,t}

Inputs:
  data/processed/eurostat_panel.csv   (from 16_eurostat_validation)
  ../paper4-lstm-agent/data/processed/features.csv
Outputs:
  results/table_panel_regression.csv
  results/figures/fig_panel_regression.{pdf,png}
"""

import os
import io
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROC_DIR = "../data/processed/"
RES_DIR  = "../results/"
FIG_DIR  = "../results/figures/"
P4_FEAT  = "../../paper4-lstm-agent/data/processed/features.csv"

print("=" * 64)
print("Paper 5 -- B2b: Cross-country panel regression")
print("=" * 64)


# ---------------------------------------------------------------------------
# 1. LOAD MONTHLY PANEL
# ---------------------------------------------------------------------------
panel_path = os.path.join(PROC_DIR, "eurostat_panel.csv")
if not os.path.exists(panel_path):
    print(f"  [error] {panel_path} not found. Run 16 first.")
    sys.exit(1)

panel = pd.read_csv(panel_path, parse_dates=["date"])
panel = panel.dropna(subset=["log_return"])
panel["year"] = panel["date"].dt.year
print(f"  monthly panel: {len(panel):,} obs, "
      f"{panel['country'].nunique()} countries")


# ---------------------------------------------------------------------------
# 2. ATTACH MATERIAL VOLATILITY (monthly)
# ---------------------------------------------------------------------------
feat = pd.read_csv(P4_FEAT, parse_dates=["Date"]).set_index("Date").sort_index()
vol_cols = [c for c in feat.columns if "vol6" in c.lower()]
feat["mat_vol"] = feat[vol_cols].mean(axis=1)
mat_vol = feat["mat_vol"].reset_index().rename(
    columns={"Date": "date", "mat_vol": "mat_vol"})

panel = panel.merge(mat_vol, on="date", how="left")
panel = panel.dropna(subset=["mat_vol"])
print(f"  panel after merging vol: {len(panel):,} obs")

# Standardise mat_vol for interpretable beta
panel["mat_vol_z"] = (panel["mat_vol"] - panel["mat_vol"].mean()) \
                       / panel["mat_vol"].std()


# ---------------------------------------------------------------------------
# 3. PANEL REGRESSIONS: pooled, country FE, country+year FE
# ---------------------------------------------------------------------------
print("\n[3] Panel regressions:")

results_rows = []

# Spec 1: pooled OLS
m1 = smf.ols("log_return ~ mat_vol_z", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["country"]})
print(f"\n  (1) pooled OLS:        beta={m1.params['mat_vol_z']:+.5f}  "
      f"p={m1.pvalues['mat_vol_z']:.4f}  R^2={m1.rsquared:.4f}")
results_rows.append({
    "spec":    "pooled OLS",
    "beta":    float(m1.params["mat_vol_z"]),
    "se":      float(m1.bse["mat_vol_z"]),
    "p":       float(m1.pvalues["mat_vol_z"]),
    "n":       int(m1.nobs),
    "r2":      float(m1.rsquared),
})

# Spec 2: country FE (within transformation)
m2 = smf.ols("log_return ~ mat_vol_z + C(country)",
             data=panel).fit(cov_type="cluster",
                              cov_kwds={"groups": panel["country"]})
print(f"  (2) country FE:        beta={m2.params['mat_vol_z']:+.5f}  "
      f"p={m2.pvalues['mat_vol_z']:.4f}  R^2={m2.rsquared:.4f}")
results_rows.append({
    "spec":    "country FE",
    "beta":    float(m2.params["mat_vol_z"]),
    "se":      float(m2.bse["mat_vol_z"]),
    "p":       float(m2.pvalues["mat_vol_z"]),
    "n":       int(m2.nobs),
    "r2":      float(m2.rsquared),
})

# Spec 3: country + year FE (two-way)
m3 = smf.ols("log_return ~ mat_vol_z + C(country) + C(year)",
             data=panel).fit(cov_type="cluster",
                              cov_kwds={"groups": panel["country"]})
print(f"  (3) country + year FE: beta={m3.params['mat_vol_z']:+.5f}  "
      f"p={m3.pvalues['mat_vol_z']:.4f}  R^2={m3.rsquared:.4f}")
results_rows.append({
    "spec":    "country + year FE",
    "beta":    float(m3.params["mat_vol_z"]),
    "se":      float(m3.bse["mat_vol_z"]),
    "p":       float(m3.pvalues["mat_vol_z"]),
    "n":       int(m3.nobs),
    "r2":      float(m3.rsquared),
})


# Spec 4: lagged volatility (predictive)
panel = panel.sort_values(["country", "date"])
panel["mat_vol_z_lag1"] = panel.groupby("country")["mat_vol_z"].shift(1)
panel_lag = panel.dropna(subset=["mat_vol_z_lag1"])
m4 = smf.ols("log_return ~ mat_vol_z_lag1 + C(country) + C(year)",
             data=panel_lag).fit(cov_type="cluster",
                                  cov_kwds={"groups": panel_lag["country"]})
print(f"  (4) lagged FE (1 mo):  beta={m4.params['mat_vol_z_lag1']:+.5f}  "
      f"p={m4.pvalues['mat_vol_z_lag1']:.4f}  R^2={m4.rsquared:.4f}")
results_rows.append({
    "spec":    "lagged (1mo) + 2-way FE",
    "beta":    float(m4.params["mat_vol_z_lag1"]),
    "se":      float(m4.bse["mat_vol_z_lag1"]),
    "p":       float(m4.pvalues["mat_vol_z_lag1"]),
    "n":       int(m4.nobs),
    "r2":      float(m4.rsquared),
})


# ---------------------------------------------------------------------------
# 4. SAVE RESULTS
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results_rows)
res_df.to_csv(os.path.join(RES_DIR, "table_panel_regression.csv"),
              index=False)
print(f"\n  Saved: table_panel_regression.csv")


# ---------------------------------------------------------------------------
# 5. FIGURE: coefficient plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(res_df))
betas = res_df["beta"].values * 100  # convert to percentage points
ses   = res_df["se"].values * 100
ax.errorbar(betas, y, xerr=1.96 * ses, fmt="o",
            markersize=10, capsize=6, color="#1976D2", lw=2)
ax.axvline(0, color="black", lw=1)
ax.set_yticks(y)
ax.set_yticklabels(res_df["spec"])
ax.set_xlabel(r"$\hat{\beta}$ on standardised mat-vol "
              r"(percentage points per 1$\sigma$)")
ax.set_title("Cross-country panel: effect of "
             "material-price volatility on construction-cost log-return\n"
             "(95% CI from country-clustered SEs)")
ax.grid(alpha=0.3, axis="x")

for i, row in res_df.iterrows():
    sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 \
          else "*" if row["p"] < 0.05 else ""
    ax.text(row["beta"] * 100 + 1.96 * row["se"] * 100 + 0.005, i,
            f"  beta={row['beta']*100:+.3f}  p={row['p']:.3f}{sig}  "
            f"n={int(row['n']):,}",
            va="center", fontsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_panel_regression.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig_panel_regression.{{pdf,png}}")

print("\n" + "=" * 64)
print("DONE -- 17_panel_regression.py")
print("=" * 64)
