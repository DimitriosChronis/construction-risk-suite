"""
Paper 4 -- SHAP Explainability (v2 -- ensemble + feature selection output)
============================================================================
Explains WHICH US PPI features drive the LSTM crisis predictions
using SHAP (SHapley Additive exPlanations).

This is the XAI contribution of Paper 4:
- Feature importance ranking
- Saves top-14 SHAP features for downstream scripts
- Creates features_selected.csv (resolves circular dependency with 01)
- Crisis vs stable SHAP distributions
- Group-level analysis (by feature type and US series)

v2 changes:
- Uses utils.py (ensemble)
- Ensemble prediction function for SHAP
- Saves features_selected.csv here (not in 01)

Install: pip install shap

Outputs:
  - data/processed/top_shap_features.csv
  - data/processed/features_selected.csv
  - results/shap_summary.csv
  - results/fig_p4_shap.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
                   DEFAULT_SEEDS)

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
LOOKBACK      = 6
LEAD          = 2
TARGET_MAT    = "GR_Fuel_Energy"
EPOCHS        = 150
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4
PATIENCE      = 20
TRAIN_RATIO   = 0.75
N_SHAP        = 100    # background samples for SHAP
N_SELECT      = 14     # top features to select (tested cutoffs 8-20, 14 optimal)

ENSEMBLE_SEEDS = DEFAULT_SEEDS

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- SHAP Explainability (Ensemble)")
print(f"Target: {TARGET_MAT} | Lead: {LEAD}M | Seeds: {len(ENSEMBLE_SEEDS)}")
print("=" * 60)

# ==============================================================================
# 1. LOAD DATA (all 20 features -- SHAP determines which to keep)
# ==============================================================================
print("\nSTEP 1: Loading processed data")

features = pd.read_csv(PROCESSED_DIR + "features.csv",
                       index_col=0, parse_dates=True)
labels   = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                       index_col=0, parse_dates=True)

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values
dates     = common

print(f"  Features : {len(FEAT_COLS)}")
print(f"  Samples  : {len(X_raw)}")

# Scale -- fit on training data only
scaler = MinMaxScaler()
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, LOOKBACK, LEAD, dates=dates)
n_train = int(len(X_seq) * TRAIN_RATIO)
X_train, X_test = X_seq[:n_train], X_seq[n_train:]
y_train, y_test = y_seq[:n_train], y_seq[n_train:]
d_test          = d_seq[n_train:]

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ==============================================================================
# 2. TRAIN ENSEMBLE
# ==============================================================================
print(f"\nSTEP 2: Training {len(ENSEMBLE_SEEDS)}-seed ensemble")

models = train_ensemble(X_train, y_train, seeds=ENSEMBLE_SEEDS,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                        dropout=DROPOUT, lr=LR, patience=PATIENCE,
                        verbose=True)

# ==============================================================================
# 3. SHAP COMPUTATION
# ==============================================================================
print("\nSTEP 3: Computing SHAP values")

# Wrapper: SHAP needs a function that takes 2D input
# We reshape: (n_samples, lookback*n_features) -> (n_samples, lookback, n_features)
def model_predict(X_flat):
    """Predict probabilities from flattened input using ensemble."""
    X_3d = X_flat.reshape(-1, LOOKBACK, len(FEAT_COLS))
    probs = predict_ensemble(models, X_3d)
    return probs.reshape(-1, 1)

# Flatten sequences for SHAP
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

# Background: random sample from training
bg_idx  = np.random.choice(len(X_train_flat), N_SHAP, replace=False)
X_bg    = X_train_flat[bg_idx]

print(f"  Background samples: {N_SHAP}")
print(f"  Test samples to explain: {len(X_test_flat)}")
print(f"  Computing KernelSHAP (this may take 2-5 minutes)...")

explainer   = shap.KernelExplainer(model_predict, X_bg)
shap_values = explainer.shap_values(X_test_flat, nsamples=200)

print(f"  SHAP values shape: {shap_values.shape}")

# ==============================================================================
# 4. AGGREGATE SHAP BY FEATURE
# ==============================================================================
print("\nSTEP 4: Aggregating SHAP by feature")

# shap_values shape: (n_test, lookback * n_features)
# Reshape to (n_test, lookback, n_features) and average over lookback
shap_3d = shap_values.reshape(len(X_test), LOOKBACK, len(FEAT_COLS))

# Mean absolute SHAP per feature (averaged over lookback and test samples)
mean_abs_shap = np.abs(shap_3d).mean(axis=(0, 1))

# Feature importance DataFrame
feat_imp = pd.DataFrame({
    "feature"   : FEAT_COLS,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 10 most important features:")
print(feat_imp.head(10).to_string(index=False))

feat_imp.to_csv(RESULTS_DIR + "shap_summary.csv", index=False)

# ==============================================================================
# 5. SAVE TOP FEATURES + FEATURES_SELECTED.CSV
# ==============================================================================
print(f"\nSTEP 5: Saving top-{N_SELECT} features and features_selected.csv")

top_features = feat_imp.head(N_SELECT)["feature"].tolist()

# Save feature names
pd.DataFrame({"feature": top_features}).to_csv(
    PROCESSED_DIR + "top_shap_features.csv", index=False)
print(f"  Saved: {PROCESSED_DIR}top_shap_features.csv")
print(f"  Features: {top_features}")

# NOTE: features_selected.csv no longer created -- all 20 features used downstream
# (ablation showed full feature set AUC=0.926 vs selected AUC=0.795)

# -- Group by feature type -------------------------------------------------
def get_group(feat):
    if "_vol6" in feat: return "6M Volatility"
    if "_vol3" in feat: return "3M Volatility"
    if "_mom3" in feat: return "3M Momentum"
    if "_ret"  in feat: return "Log Return"
    return "Other"

def get_series(feat):
    for s in ["Brent", "Steel_PPI", "Cement_PPI", "PVC_PPI", "Fuel_PPI"]:
        if s in feat:
            return s.replace("_PPI", "")
    return feat

feat_imp["group"]  = feat_imp["feature"].apply(get_group)
feat_imp["series"] = feat_imp["feature"].apply(get_series)

group_imp  = feat_imp.groupby("group")["mean_abs_shap"].sum().sort_values(
    ascending=False)
series_imp = feat_imp.groupby("series")["mean_abs_shap"].sum().sort_values(
    ascending=False)

print("\nImportance by feature type:")
print(group_imp.to_string())
print("\nImportance by US series:")
print(series_imp.to_string())

# -- SHAP per class --------------------------------------------------------
shap_crisis = shap_3d[y_test == 1].mean(axis=(0, 1))
shap_stable = shap_3d[y_test == 0].mean(axis=(0, 1))

class_df = pd.DataFrame({
    "feature"     : FEAT_COLS,
    "shap_crisis" : shap_crisis,
    "shap_stable" : shap_stable,
    "difference"  : shap_crisis - shap_stable
}).sort_values("difference", ascending=False)

print("\nFeatures with highest crisis vs stable SHAP difference:")
print(class_df.head(10).to_string(index=False))

# ==============================================================================
# 6. FIGURES
# ==============================================================================
print("\nSTEP 6: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    f"Paper 4 -- SHAP Explainability (Ensemble)\n"
    f"US PPI -> {TARGET_MAT} Crisis (lead={LEAD}M)",
    fontsize=13, fontweight="bold"
)

# Panel 1: Top 15 feature importance bar chart
ax = axes[0, 0]
top15 = feat_imp.head(15)
colors = []
for feat in top15["feature"]:
    if "Fuel" in feat: colors.append("#e74c3c")
    elif "Steel" in feat: colors.append("#3498db")
    elif "Cement" in feat: colors.append("#2ecc71")
    elif "PVC" in feat: colors.append("#9b59b6")
    else: colors.append("#f39c12")

bars = ax.barh(range(len(top15)), top15["mean_abs_shap"],
               color=colors, edgecolor="black", linewidth=0.3)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f.replace("US_", "").replace("_PPI", "")
                    for f in top15["feature"]], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top 15 Feature Importance (SHAP)\nTop-{N_SELECT} selected for downstream")
ax.grid(alpha=0.3, axis="x")

# Highlight selected features
for i, feat in enumerate(top15["feature"]):
    if feat in top_features:
        bars[i].set_edgecolor("red")
        bars[i].set_linewidth(1.5)

legend_elements = [
    Patch(facecolor="#e74c3c", label="Fuel PPI"),
    Patch(facecolor="#3498db", label="Steel PPI"),
    Patch(facecolor="#2ecc71", label="Cement PPI"),
    Patch(facecolor="#9b59b6", label="PVC PPI"),
    Patch(facecolor="#f39c12", label="Brent"),
]
ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

# Panel 2: Importance by group (feature type)
ax = axes[0, 1]
group_colors = {"6M Volatility": "#e74c3c", "3M Volatility": "#e67e22",
                "3M Momentum": "#3498db", "Log Return": "#2ecc71"}
bars2 = ax.bar(range(len(group_imp)), group_imp.values,
               color=[group_colors.get(g, "#95a5a6")
                      for g in group_imp.index],
               edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, group_imp.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + group_imp.values.max() * 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(group_imp)))
ax.set_xticklabels(group_imp.index, rotation=15, fontsize=9)
ax.set_ylabel("Total |SHAP|")
ax.set_title("Feature Importance by Type\n"
             "(Volatility features dominate -- regime-sensitive signals)")
ax.grid(alpha=0.3, axis="y")

# Panel 3: Importance by US series
ax = axes[1, 0]
series_colors = {"Fuel": "#e74c3c", "Steel": "#3498db",
                 "Cement": "#2ecc71", "PVC": "#9b59b6", "Brent": "#f39c12"}
bars3 = ax.bar(range(len(series_imp)), series_imp.values,
               color=[series_colors.get(s, "#95a5a6")
                      for s in series_imp.index],
               edgecolor="black", linewidth=0.5)
for bar, val in zip(bars3, series_imp.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + series_imp.values.max() * 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(series_imp)))
ax.set_xticklabels(series_imp.index, fontsize=10)
ax.set_ylabel("Total |SHAP|")
ax.set_title("Feature Importance by US Series\n"
             "(Which US commodity drives Greek Fuel crisis?)")
ax.grid(alpha=0.3, axis="y")

# Panel 4: Crisis vs Stable SHAP comparison (top 10)
ax = axes[1, 1]
top10_diff = class_df.head(10)
x = np.arange(len(top10_diff))
w = 0.35
ax.bar(x - w/2, top10_diff["shap_crisis"], w,
       label="Crisis months", color="#e74c3c", alpha=0.8,
       edgecolor="black", linewidth=0.3)
ax.bar(x + w/2, top10_diff["shap_stable"], w,
       label="Stable months", color="#3498db", alpha=0.8,
       edgecolor="black", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(
    [f.replace("US_", "").replace("_PPI", "")
     for f in top10_diff["feature"]],
    rotation=30, fontsize=7, ha="right")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Mean SHAP value")
ax.set_title("SHAP: Crisis vs Stable months\n"
             "(positive = pushes toward crisis prediction)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_shap.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

# ==============================================================================
# 7. KEY FINDING SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("KEY FINDINGS FOR PAPER 4:")
print("=" * 60)
top_feat   = feat_imp.iloc[0]
top_series = series_imp.index[0]
top_group  = group_imp.index[0]

print(f"\n1. Most important feature: {top_feat['feature']}")
print(f"   Mean |SHAP|: {top_feat['mean_abs_shap']:.5f}")
print(f"\n2. Most important US series: {top_series}")
print(f"   Total SHAP: {series_imp.iloc[0]:.5f}")
print(f"\n3. Most important feature type: {top_group}")
print(f"   Total SHAP: {group_imp.iloc[0]:.5f}")
print(f"\n4. Top-{N_SELECT} features identified (SHAP ranking)")
print(f"   All 20 features used downstream (no filtering)")

print("\n" + "=" * 60)
print("DONE -- SHAP explainability complete.")
print("Next: run 04_walk_forward_validation.py")
print("=" * 60)
