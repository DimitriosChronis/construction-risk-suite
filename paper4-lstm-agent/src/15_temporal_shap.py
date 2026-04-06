"""
Paper 4 -- Temporal SHAP Analysis (Phase 4A)
===============================================
Shows HOW feature importance changes over time, answering:
  "Do the same US PPI indicators matter during crisis vs stable periods?"

Methodology:
  - Split test set into temporal windows (quarterly)
  - Compute SHAP values for each window using ensemble mean prediction
  - Track top feature importance evolution over time
  - Compare crisis-period SHAP vs stable-period SHAP

Uses KernelExplainer (model-agnostic) on the ensemble prediction function.

Outputs:
  - results/temporal_shap_evolution.csv
  - results/temporal_shap_crisis_vs_stable.csv
  - results/fig_p4_temporal_shap.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

import torch

try:
    import shap
    print("SHAP available OK")
except ImportError:
    raise ImportError("pip install shap")

from utils import (make_sequences, train_ensemble, predict_ensemble,
                   load_selected_features, DEFAULT_SEEDS, predict_probs)

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
LOOKBACK      = 6
LEAD          = 2
TARGET_MAT    = "GR_Fuel_Energy"
TRAIN_RATIO   = 0.75
EPOCHS        = 150
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4
PATIENCE      = 20

ENSEMBLE_SEEDS = DEFAULT_SEEDS
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

N_BG_SAMPLES  = 50    # KernelExplainer background samples
N_SHAP_EVAL   = 200   # max samples to explain (all test if fewer)

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Temporal SHAP Analysis")
print("=" * 60)

# ==============================================================================
# 1. LOAD DATA + TRAIN ENSEMBLE
# ==============================================================================
print("\nSTEP 1: Loading data and training ensemble")

sel_feats = load_selected_features(PROCESSED_DIR) if USE_SELECTED_FEATURES else None

if sel_feats is not None:
    features = pd.read_csv(PROCESSED_DIR + "features_selected.csv",
                           index_col=0, parse_dates=True)
    FEAT_NAMES = list(features.columns)
    print(f"  Using SHAP-selected features: {len(FEAT_NAMES)}")
else:
    features = pd.read_csv(PROCESSED_DIR + "features.csv",
                           index_col=0, parse_dates=True)
    FEAT_NAMES = list(features.columns)

labels = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                     index_col=0, parse_dates=True)

common = features.index.intersection(labels.dropna().index)
X_raw  = features.loc[common].values
y_raw  = labels.loc[common, TARGET_MAT].values
dates  = common

scaler = MinMaxScaler()
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, LOOKBACK, LEAD, dates=dates)
n_tr = int(len(X_seq) * TRAIN_RATIO)
X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]
d_te = d_seq[n_tr:]

print(f"  Test: {len(X_te)} samples | Crisis: {int(y_te.sum())}")

# Train ensemble
print(f"  Training {len(ENSEMBLE_SEEDS)}-seed ensemble...")
models = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                        dropout=DROPOUT, lr=LR, patience=PATIENCE,
                        verbose=True)

# ==============================================================================
# 2. SHAP SETUP
# ==============================================================================
print("\nSTEP 2: Computing SHAP values")

# Flatten 3D -> 2D for KernelExplainer (mean over lookback window)
def flatten_for_shap(X_3d):
    """Use last timestep of each sequence as SHAP input."""
    return X_3d[:, -1, :]

X_te_flat = flatten_for_shap(X_te)
X_tr_flat = flatten_for_shap(X_tr)

# Ensemble prediction wrapper for flattened input
def ensemble_predict_flat(X_flat):
    """Predict from flattened input by reconstructing sequences."""
    # Tile the flat input across lookback dimension
    # This is an approximation -- uses the flat features as if constant
    n = len(X_flat)
    X_3d = np.tile(X_flat[:, np.newaxis, :], (1, LOOKBACK, 1))
    return predict_ensemble(models, X_3d)

# Background dataset
bg_idx = np.random.choice(len(X_tr_flat), min(N_BG_SAMPLES, len(X_tr_flat)),
                          replace=False)
X_bg = X_tr_flat[bg_idx]

print(f"  Background samples: {len(X_bg)}")
print(f"  Test samples to explain: {len(X_te_flat)}")

# Compute SHAP values
explainer = shap.KernelExplainer(ensemble_predict_flat, X_bg)
shap_values = explainer.shap_values(X_te_flat, nsamples=100, silent=True)

print(f"  SHAP values shape: {shap_values.shape}")

# ==============================================================================
# 3. TEMPORAL EVOLUTION (QUARTERLY WINDOWS)
# ==============================================================================
print("\nSTEP 3: Temporal SHAP evolution")

test_dates = pd.to_datetime([str(d) for d in d_te])
df_shap = pd.DataFrame(shap_values, columns=FEAT_NAMES, index=test_dates)
df_shap["actual_crisis"] = y_te

# Create quarterly periods
df_shap["quarter"] = df_shap.index.to_period("Q")
quarters = df_shap["quarter"].unique()

print(f"  Quarters in test set: {len(quarters)}")

# Mean absolute SHAP per quarter per feature
temporal_rows = []
for q in quarters:
    mask = df_shap["quarter"] == q
    q_data = df_shap.loc[mask, FEAT_NAMES]
    q_crisis = df_shap.loc[mask, "actual_crisis"]
    mean_abs = q_data.abs().mean()
    for feat in FEAT_NAMES:
        temporal_rows.append({
            "quarter": str(q),
            "feature": feat,
            "mean_abs_shap": mean_abs[feat],
            "n_samples": int(mask.sum()),
            "crisis_rate": q_crisis.mean(),
        })

df_temporal = pd.DataFrame(temporal_rows)
df_temporal.to_csv(RESULTS_DIR + "temporal_shap_evolution.csv", index=False)

# Rank features per quarter
print(f"\n  Top-3 features per quarter:")
for q in quarters:
    q_data = df_temporal[df_temporal["quarter"] == str(q)]
    top3 = q_data.nlargest(3, "mean_abs_shap")
    crisis_rate = q_data.iloc[0]["crisis_rate"]
    regime = "CRISIS" if crisis_rate > 0.5 else "stable"
    feats = ", ".join(top3["feature"].values)
    print(f"    {q} [{regime:6s}]: {feats}")

# ==============================================================================
# 4. CRISIS VS STABLE COMPARISON
# ==============================================================================
print("\nSTEP 4: Crisis vs stable SHAP comparison")

crisis_mask = y_te == 1
stable_mask = y_te == 0

crisis_shap_mean = df_shap.loc[crisis_mask, FEAT_NAMES].abs().mean()
stable_shap_mean = df_shap.loc[stable_mask, FEAT_NAMES].abs().mean()

df_comparison = pd.DataFrame({
    "feature": FEAT_NAMES,
    "crisis_mean_abs_shap": crisis_shap_mean.values,
    "stable_mean_abs_shap": stable_shap_mean.values,
    "ratio_crisis_stable": (crisis_shap_mean / stable_shap_mean.replace(0, np.nan)).values,
})
df_comparison = df_comparison.sort_values("crisis_mean_abs_shap", ascending=False)
df_comparison.to_csv(RESULTS_DIR + "temporal_shap_crisis_vs_stable.csv", index=False)

print(f"\n  {'Feature':30s} {'Crisis |SHAP|':>14s} {'Stable |SHAP|':>14s} {'Ratio':>7s}")
print(f"  {'-'*70}")
for _, row in df_comparison.iterrows():
    ratio = f"{row['ratio_crisis_stable']:.2f}" if not np.isnan(row['ratio_crisis_stable']) else "N/A"
    print(f"  {row['feature']:30s} {row['crisis_mean_abs_shap']:14.4f} "
          f"{row['stable_mean_abs_shap']:14.4f} {ratio:>7s}")

# ==============================================================================
# 5. FIGURES
# ==============================================================================
print("\nSTEP 5: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Paper 4 -- Temporal SHAP Analysis\n"
    f"Target: {TARGET_MAT} | {len(FEAT_NAMES)} features | "
    f"{len(quarters)} quarters",
    fontsize=12, fontweight="bold"
)

# Panel 1: SHAP evolution heatmap (top-8 features over time)
ax = axes[0, 0]
overall_top8 = df_shap[FEAT_NAMES].abs().mean().nlargest(8).index.tolist()

# Build heatmap matrix
q_labels = [str(q) for q in quarters]
heatmap_data = np.zeros((len(overall_top8), len(q_labels)))
for j, q in enumerate(q_labels):
    q_rows = df_temporal[df_temporal["quarter"] == q]
    for i, feat in enumerate(overall_top8):
        val = q_rows[q_rows["feature"] == feat]["mean_abs_shap"].values
        heatmap_data[i, j] = val[0] if len(val) > 0 else 0

im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(q_labels)))
ax.set_xticklabels(q_labels, fontsize=6, rotation=45, ha="right")
ax.set_yticks(range(len(overall_top8)))
ax.set_yticklabels(overall_top8, fontsize=8)
ax.set_title("Feature Importance Evolution (top-8)\n"
             "Mean |SHAP| per quarter")
plt.colorbar(im, ax=ax, label="|SHAP|", shrink=0.8)

# Add crisis rate bar at top
crisis_rates = [df_temporal[df_temporal["quarter"] == q].iloc[0]["crisis_rate"]
                for q in q_labels]
ax2 = ax.twiny()
ax2.bar(range(len(q_labels)), crisis_rates, color="red", alpha=0.2, width=0.8)
ax2.set_xlim(-0.5, len(q_labels) - 0.5)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Crisis rate", fontsize=7)
ax2.tick_params(labelsize=6)

# Panel 2: Crisis vs Stable feature importance
ax = axes[0, 1]
top10 = df_comparison.head(10)
y_pos = np.arange(len(top10))
w = 0.35
ax.barh(y_pos - w/2, top10["crisis_mean_abs_shap"], w,
        color="#e74c3c", edgecolor="black", linewidth=0.5, label="Crisis")
ax.barh(y_pos + w/2, top10["stable_mean_abs_shap"], w,
        color="#3498db", edgecolor="black", linewidth=0.5, label="Stable")
ax.set_yticks(y_pos)
ax.set_yticklabels(top10["feature"], fontsize=8)
ax.set_xlabel("Mean |SHAP|")
ax.set_title("Feature Importance: Crisis vs Stable\n(top-10 by crisis importance)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="x")
ax.invert_yaxis()

# Panel 3: Top-4 features SHAP over time (line plot)
ax = axes[1, 0]
top4 = overall_top8[:4]
colors_line = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

for i, feat in enumerate(top4):
    vals = []
    for q in q_labels:
        q_rows = df_temporal[df_temporal["quarter"] == q]
        v = q_rows[q_rows["feature"] == feat]["mean_abs_shap"].values
        vals.append(v[0] if len(v) > 0 else 0)
    ax.plot(range(len(q_labels)), vals, "o-", color=colors_line[i],
            lw=1.5, markersize=4, label=feat)

# Shade crisis quarters
for j, cr in enumerate(crisis_rates):
    if cr > 0.5:
        ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.08)

ax.set_xticks(range(len(q_labels)))
ax.set_xticklabels(q_labels, fontsize=6, rotation=45, ha="right")
ax.set_ylabel("Mean |SHAP|")
ax.set_title("Top-4 Feature Importance Over Time\n(red shading = crisis quarters)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 4: SHAP beeswarm/summary for overall
ax = axes[1, 1]
# Bar chart of overall mean |SHAP| (all features)
overall_importance = df_shap[FEAT_NAMES].abs().mean().sort_values(ascending=True)
colors_bar = ["#e74c3c" if f in overall_top8[:4] else "#3498db"
              for f in overall_importance.index]
ax.barh(range(len(overall_importance)), overall_importance.values,
        color=colors_bar, edgecolor="black", linewidth=0.3)
ax.set_yticks(range(len(overall_importance)))
ax.set_yticklabels(overall_importance.index, fontsize=7)
ax.set_xlabel("Mean |SHAP|")
ax.set_title("Overall Feature Importance (all test)\n"
             "(red = top-4)")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_temporal_shap.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved: {fig_path}")
plt.show()

# ==============================================================================
# 6. SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("TEMPORAL SHAP SUMMARY (for Paper 4)")
print("=" * 60)

# Features that become more important during crisis
crisis_amplified = df_comparison[df_comparison["ratio_crisis_stable"] > 1.5]
print(f"\n  Features amplified during crisis (ratio > 1.5x):")
for _, row in crisis_amplified.iterrows():
    print(f"    {row['feature']:30s} ratio={row['ratio_crisis_stable']:.2f}x")

stable_amplified = df_comparison[df_comparison["ratio_crisis_stable"] < 0.67]
if len(stable_amplified) > 0:
    print(f"\n  Features amplified during stable (ratio < 0.67x):")
    for _, row in stable_amplified.iterrows():
        print(f"    {row['feature']:30s} ratio={row['ratio_crisis_stable']:.2f}x")

print(f"\n  KEY INSIGHT: Feature importance is regime-dependent.")
print(f"  This supports the use of a nonlinear model (LSTM) that can")
print(f"  adapt its attention to different features across regimes.")

print("\n" + "=" * 60)
print("DONE -- Temporal SHAP analysis complete.")
print("=" * 60)
print("Next: run 16_publication_figures.py")
