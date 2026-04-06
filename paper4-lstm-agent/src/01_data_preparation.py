"""
Paper 4 — Data Preparation
============================
Loads aligned log-returns from Paper 3, builds features and
crisis regime labels, and saves processed datasets for all
subsequent Paper 4 scripts.

Outputs (saved to data/processed/):
  - features.csv          : 20 engineered US PPI features
  - crisis_labels.csv     : binary crisis flag per material
  - aligned_returns.csv   : raw aligned log-returns (copy)
  - data_summary.csv      : dataset statistics

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL PARAMETERS — shared across all Paper 4 scripts
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH      = "../../paper3-es-hedging/data/processed/aligned_log_returns.csv"
OUT_DIR        = "../data/processed/"
RESULTS_DIR    = "../results/"
SEED           = 42
VOL_WINDOW     = 6       # rolling window for crisis definition
CRISIS_PCT     = 0.75    # P75 threshold
LOOKBACK       = 6       # LSTM lookback window (months)
LEAD_TIMES     = [1, 2, 3, 4]
TARGET_MATERIALS = ["GR_Fuel_Energy", "GR_Steel",
                    "GR_Concrete", "GR_PVC_Pipes"]

# Named crisis regimes (consistent with Papers 1-3)
STABLE_START  = "2014-01-01"
STABLE_END    = "2019-12-01"
CRISIS_START  = "2021-01-01"
CRISIS_END    = "2024-12-01"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)

print("=" * 60)
print("Paper 4 — Data Preparation")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading aligned log-returns")

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"  Loaded : {df.shape[0]} obs x {df.shape[1]} series")
print(f"  Period : {df.index[0].strftime('%Y-%m')} -> "
      f"{df.index[-1].strftime('%Y-%m')}")

GR_COLS = [c for c in df.columns if c.startswith("GR_")]
US_COLS = [c for c in df.columns if c.startswith("US_")]
print(f"  Greek  : {GR_COLS}")
print(f"  US     : {US_COLS}")

# Save copy
df.to_csv(OUT_DIR + "aligned_returns.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Feature engineering (US PPI signals)")

feat_parts = []

# (a) Raw log-returns
ret_df = df[US_COLS].rename(
    columns={c: f"{c}_ret" for c in US_COLS})
feat_parts.append(ret_df)

# (b) Rolling 3M volatility
for c in US_COLS:
    feat_parts.append(
        df[c].rolling(3).std().rename(f"{c}_vol3"))

# (c) Rolling 6M volatility
for c in US_COLS:
    feat_parts.append(
        df[c].rolling(6).std().rename(f"{c}_vol6"))

# (d) 3M momentum
for c in US_COLS:
    feat_parts.append(
        df[c].rolling(3).sum().rename(f"{c}_mom3"))

features = pd.concat(feat_parts, axis=1).dropna()
FEAT_COLS = list(features.columns)

print(f"  Features : {len(FEAT_COLS)} columns")
print(f"  Period   : {features.index[0].strftime('%Y-%m')} -> "
      f"{features.index[-1].strftime('%Y-%m')}")
print(f"  Columns  : {FEAT_COLS}")

features.to_csv(OUT_DIR + "features.csv")

# NOTE: features_selected.csv is created by 03_shap_explanations.py (SHAP-based selection)

# ══════════════════════════════════════════════════════════════════════════════
# 3. CRISIS LABELS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Crisis regime labels")

labels = pd.DataFrame(index=df.index)
thresholds = {}
summary_rows = []

for mat in TARGET_MATERIALS:
    vol = df[mat].rolling(VOL_WINDOW).std()
    thr = vol.quantile(CRISIS_PCT)
    thresholds[mat] = thr
    labels[mat] = (vol > thr).astype(int)

    n_total  = labels[mat].notna().sum()
    n_crisis = int(labels[mat].sum())
    pct      = n_crisis / n_total * 100

    # Regime-specific stats
    stable_crisis = labels[mat].loc[STABLE_START:STABLE_END].mean()
    crisis_crisis = labels[mat].loc[CRISIS_START:CRISIS_END].mean()

    print(f"  {mat}:")
    print(f"    Threshold (P75) : {thr:.4f}")
    print(f"    Crisis months   : {n_crisis}/{n_total} ({pct:.1f}%)")
    print(f"    Crisis rate — stable regime  : {stable_crisis*100:.1f}%")
    print(f"    Crisis rate — crisis regime  : {crisis_crisis*100:.1f}%")

    summary_rows.append({
        "material"         : mat,
        "threshold_P75"    : thr,
        "n_crisis"         : n_crisis,
        "n_total"          : n_total,
        "pct_crisis"       : pct,
        "stable_crisis_rate": stable_crisis,
        "crisis_crisis_rate": crisis_crisis
    })

labels.to_csv(OUT_DIR + "crisis_labels.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LEAD-LAG CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Lead-lag correlation (US features -> Greek crisis)")

corr_rows = []
common_idx = features.index.intersection(labels.dropna().index)

for mat in TARGET_MATERIALS:
    for feat in FEAT_COLS:
        for lag in LEAD_TIMES:
            feat_lagged = features.loc[common_idx, feat].shift(lag)
            lbl = labels.loc[common_idx, mat]
            corr = feat_lagged.corr(lbl)
            corr_rows.append({
                "material": mat,
                "feature" : feat,
                "lead"    : lag,
                "corr"    : corr
            })

df_corr = pd.DataFrame(corr_rows)

# Top correlations per material
print("\n  Top 5 features per material (best lead):")
for mat in TARGET_MATERIALS:
    top = (df_corr[df_corr["material"] == mat]
           .sort_values("corr", ascending=False)
           .head(5))
    print(f"\n  {mat}:")
    print(top[["feature", "lead", "corr"]].to_string(index=False))

df_corr.to_csv(OUT_DIR + "leadlag_correlations.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Saving summary")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR + "data_summary.csv", index=False)

print(f"""
  Output files saved to {OUT_DIR}:
    - aligned_returns.csv
    - features.csv
    - crisis_labels.csv
    - leadlag_correlations.csv
    - data_summary.csv
""")

# ══════════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("STEP 6: Diagnostic figure")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Paper 4 — Data Overview: US PPI Features & Crisis Regimes",
             fontsize=12, fontweight="bold")

# Panel 1: US series log-returns
ax = axes[0, 0]
for c in US_COLS:
    ax.plot(df.index, df[c], lw=0.8, alpha=0.7,
            label=c.replace("US_", ""))
ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
           color="red", alpha=0.1, label="Crisis regime")
ax.set_title("US PPI Log-Returns (2000–2024)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 2: Greek rolling volatility
ax = axes[0, 1]
for mat in TARGET_MATERIALS:
    vol = df[mat].rolling(VOL_WINDOW).std()
    ax.plot(vol.index, vol, lw=0.9, alpha=0.8,
            label=mat.replace("GR_", ""))
ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
           color="red", alpha=0.1)
ax.set_title(f"Greek Material Rolling Volatility ({VOL_WINDOW}M)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Panel 3: Crisis flags timeline
ax = axes[1, 0]
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
for i, mat in enumerate(TARGET_MATERIALS):
    crisis = labels[mat].dropna()
    ax.fill_between(crisis.index,
                    i + crisis.values * 0.8,
                    i,
                    color=colors[i], alpha=0.6,
                    label=mat.replace("GR_", ""))
ax.set_yticks(range(len(TARGET_MATERIALS)))
ax.set_yticklabels([m.replace("GR_", "") for m in TARGET_MATERIALS])
ax.set_title("Crisis Regime Flags per Material (P75 threshold)")
ax.grid(alpha=0.3)

# Panel 4: Top lead-lag correlations
ax = axes[1, 1]
fuel_corr = (df_corr[df_corr["material"] == "GR_Fuel_Energy"]
             .groupby(["feature", "lead"])["corr"]
             .mean().reset_index())
top_feats = (fuel_corr.sort_values("corr", ascending=False)
             .head(20)["feature"].unique()[:5])
for feat in top_feats:
    sub = fuel_corr[fuel_corr["feature"] == feat].sort_values("lead")
    ax.plot(sub["lead"], sub["corr"], marker="o",
            label=feat.replace("US_", ""))
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Lead time (months)")
ax.set_ylabel("Correlation with Fuel crisis flag")
ax.set_title("Top US Features -> GR_Fuel_Energy (lead-lag)")
ax.legend(fontsize=7)
ax.set_xticks(LEAD_TIMES)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_data_overview.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved: {fig_path}")
plt.show()

print("\n" + "=" * 60)
print("DONE — Data preparation complete.")
print("Next: run 02_lstm_regime_classification.py")
print("=" * 60)